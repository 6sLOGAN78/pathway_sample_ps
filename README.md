# Multi-Agent AI-Powered Customer Support System

## Overview

This project implements a **multi-agent AI-powered customer support system** capable of handling user queries in **text, PDF, DOCX, and image formats**. It leverages **Google Gemini 2.5-Flash**, **LangChain**, **retrieval-augmented generation (RAG)**, and **persistent user memory** to provide context-aware, personalized responses.  

The system dynamically decides how to respond using specialized agents and stores user memory to improve interaction over time.

---

## Features & Workflow

1. **Multi-Modal Input Handling**
   - Accepts **text queries**, **PDFs**, **DOCX files**, and **images** (PNG/JPG).
   - Converts documents into **bytes** for Gemini processing.
   - Generates **embeddings** for large files for later retrieval via **RAG**.
   - Extracts images inside PDFs for embedding and context retrieval.

2. **Decision Agent**
   - Determines if a query requires:
     - Web search
     - Memory update
     - Reference to previously uploaded documents
   - Routes queries to the appropriate agent.

3. **Document & RAG Processing**
   - Splits large documents into chunks.
   - Embeds text using **SentenceTransformer**.
   - Embeds images using **CLIP**.
   - Stores embeddings in **ChromaDB** for RAG retrieval.

4. **Main Conversational Agent**
   - Powered by **Gemini 2.5-Flash** through **LangChain**.
   - Receives user input, context from RAG, and memory data.
   - Generates personalized, context-aware responses.

5. **User Memory & Personalization**
   - Persistent memory stored in `user_details.txt`.
   - **Memory Control Agent** monitors token count:
     - Compresses memory if it exceeds **2000 tokens** using a financial-analytic AI.
   - Ensures memory is always efficient and relevant.

6. **Web Search Agent**
   - Performs live searches when needed.
   - Returns summarized content and verified sources.

7. **Conversation Management**
   - Stores conversations in **JSON** files.
   - Allows resuming previous sessions.
   - Maintains references to uploaded documents and embeddings.

---

## File Structure & Explanation

### Core Scripts

| File | Description | Link |
|------|-------------|------|
| [main.py](./main.py) | Orchestrates the chat loop, integrates all agents, handles user input, memory, document upload, web search, and generates AI responses. Entry point of the application. | [View Code](./main.py) |
| [document_saver.py](./document_saver.py) | Handles PDF/DOCX/text extraction, image processing, embedding generation, and storing embeddings in **ChromaDB**. Supports **text chunks** and **image embeddings**. | [View Code](./document_saver.py) |
| [user_data_control.py](./user_data_control.py) | Manages **user memory** in `user_details.txt`. Monitors token count and triggers memory compression using a financial-analytic AI agent if memory exceeds 2000 tokens. | [View Code](./user_data_control.py) |
| [web_search.py](./web_search.py) | Implements **WebQuery Agent** that performs live web searches using Google Gemini API and retrieves summarized answers with sources. | [View Code](./web_search.py) |
| [evaluator.py](./evaluator.py) | Implements **DecisionAgent**. Analyzes user queries to determine whether web search is needed, memory should be updated, or document retrieval is required. | [View Code](./evaluator.py) |
| [naming_agent.py](./naming_agent.py) | Generates dynamic names for conversations based on user queries to help identify and store sessions. | [View Code](./naming_agent.py) |
[json_helper.py](./json_helper.py) | Helps to handle json file and updates. | [View Code](./json_helper.py) |
[get_news.py](./get_news.py) | takes user_details.txt and according to users likes and dislikes it updates the news section. | [View Code](./get_news.py) |
### Data & Storage

| File / Folder | Description | Link |
|---------------|-------------|------|
| `user_details.txt` | Stores persistent user memory and preferences. Used to personalize AI responses across sessions. | [View File](./user_details.txt) |
| `conversation/` | Stores **JSON conversation files**, each containing full chat history and references to uploaded documents. | [View Folder](./conversation) |
| `embeddings7/chromadb8` | Persistent storage for embeddings used in **RAG**. Text and image embeddings are saved here for retrieval. | [View Folder](./embeddings7/chromadb8) |

