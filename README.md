# Enterprise RAG Project

An enterprise-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, LangChain, FAISS, and Groq. This application allows users to upload enterprise documents, securely process them, and query them using high-speed LLMs with accurate contextual retrieval.

---

## Features

* **FastAPI Backend:** High-performance asynchronous REST API handling document parsing, embedding generation, and vector retrieval.
* **Streamlit Frontend:** Clean, interactive user interface for file uploads, system status monitoring, and real-time chat/querying.
* **LangChain & FAISS:** Efficient document chunking, vector embedding, and similarity search powered by FAISS.
* **Groq LLM Integration:** Ultra-fast inference leveraging Groq's high-speed language models for response generation.
* **Secure Configuration:** Environment-based secret management using `python-dotenv`.

---

## Project Structure

```text
Enterprise_RAG_Project/
│
├── app.py              # Streamlit frontend application
├── main.py             # FastAPI backend application
├── requirements.txt    # Project dependencies
├── .env                # Environment variables (API keys - ignored by Git)
├── .gitignore          # Git ignore file
├── faiss_index/        # Local FAISS vector store directory (ignored by Git)
└── uploads/            # Temporary document upload folder (ignored by Git)