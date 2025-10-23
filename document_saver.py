import os
import io
import fitz  
import docx
from PIL import Image
import torch
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter


PERSIST_DIR = "embeddings7/chromadb8"
CHUNKS_DIR = "chunks"
TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
IMAGE_MODEL_NAME = "openai/clip-vit-large-patch14"
MAX_WORDS_PER_CHUNK = 2000
device = "cuda" if torch.cuda.is_available() else "cpu"

#embeddings
text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device, trust_remote_code=True)
text_model.max_seq_length = 4096
text_model.eval()

clip_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
clip_model.eval()

os.makedirs(CHUNKS_DIR, exist_ok=True)

#Splitter 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_WORDS_PER_CHUNK,
    chunk_overlap=200
)

def split_text_to_chunks(text: str):
    if not text.strip():
        return []
    return splitter.split_text(text)

# Extraction 
def extract_text_from_pdf(pdf_path: str):
    txts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txts.append(page.get_text("text"))
    return txts

def extract_images_from_pdf(pdf_path: str):
    images = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                    image_id = f"{os.path.basename(pdf_path)}_page_{i}_img_{img_index}"
                    images.append((i, pil_image, image_id))
                except Exception as e:
                    print(f"[!] Error extracting image {img_index} on page {i}: {e}")
    return images

def extract_text_from_docx(path: str):
    doc = docx.Document(path)
    return [p.text for p in doc.paragraphs if p.text.strip()]

# Embeddings 
def embed_text_chunk(text: str) -> np.ndarray:
    if not text.strip():
        return None
    emb = text_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().numpy()

def embed_image(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

# database
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection_name = "multimodal_embeddings"
if collection_name in [c.name for c in client.list_collections()]:
    col = client.get_collection(collection_name)
else:
    col = client.create_collection(name=collection_name)

def store_in_chroma(ids, embeddings, metadatas, documents):
    if not embeddings:
        print("[WARN] No embeddings to store, skipping.")
        return
    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

# Main
def save_documents_for_future(documents):
    for doc in documents:
        doc_path = doc["path"]
        doc_name = os.path.basename(doc_path)
        embeddings = []
        ids = []
        metadatas = []
        docs_text = []

        # Text 
        text_pages = []
        if doc_path.lower().endswith(".pdf"):
            text_pages = extract_text_from_pdf(doc_path)
        elif doc_path.lower().endswith(".docx"):
            text_pages = extract_text_from_docx(doc_path)

        for i, page_text in enumerate(text_pages):
            chunks = split_text_to_chunks(page_text)
            for j, chunk in enumerate(chunks):
                emb = embed_text_chunk(chunk)
                if emb is not None:
                    embeddings.append(emb)
                    ids.append(f"{doc_name}_text_{i}_{j}")
                    metadatas.append({"type": "text", "source": doc_name, "page": i})
                    docs_text.append(chunk)

        # Images in PDFs
        if doc_path.lower().endswith(".pdf"):
            images = extract_images_from_pdf(doc_path)
            for i, img, img_id in images:
                emb = embed_image(img)
                embeddings.append(emb)
                ids.append(img_id)
                metadatas.append({"type": "image", "source": doc_name, "page": i})
                docs_text.append("")

        # Images 
        if doc_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(doc_path).convert("RGB")
                emb = embed_image(img)
                embeddings.append(emb)
                ids.append(doc_name)
                metadatas.append({"type": "image", "source": doc_name})
                docs_text.append("")
                print(f"[INFO] Embedded standalone image: {doc_name}")
            except Exception as e:
                print(f"[ERROR] Failed to embed image {doc_name}: {e}")

        # Store in Chroma 
        store_in_chroma(ids, embeddings, metadatas, docs_text)
