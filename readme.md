Here’s a clean, professional **`README.md`** you can include with your FastAPI + Ollama + Google OAuth backend 👇

---

````markdown
# 🚀 FastAPI + Google OAuth + Ollama + ChromaDB Backend

This project connects **Google APIs (Gmail & Drive)** with a **local Ollama LLM** and **ChromaDB** to enable retrieval-augmented Q&A over your personal data.

---

## 🧩 Features

- ✅ Google OAuth2 login (Gmail + Drive)
- 📧 Fetch and index Gmail messages
- 📁 Fetch and index Google Drive files
- 🧠 Embed data into **ChromaDB** using **Ollama embeddings**
- 💬 Ask natural-language questions using **Retrieval-Augmented Generation (RAG)**
- 🧰 Built with **FastAPI**, **ChromaDB**, **Ollama**, and **Google API client**

---

## 📦 Requirements

### 1. System Requirements
- Python **3.10+**
- Ollama installed locally → [Install here](https://ollama.ai)
- Installed Ollama models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3
````

(Optional fallback models)

```bash
ollama pull bge-m3
ollama pull mxbai-embed-large
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fastapi-ollama-rag.git
cd fastapi-ollama-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, you can generate one with:

```bash
pip install fastapi uvicorn python-dotenv requests google-auth google-auth-oauthlib google-api-python-client chromadb
pip freeze > requirements.txt
```

---

## 🔐 Google OAuth Setup

### 1. Go to [Google Cloud Console](https://console.cloud.google.com/)

* Create a new project
* Enable APIs:

  * **Gmail API**
  * **Google Drive API**
* Go to **APIs & Services → Credentials**

  * Create **OAuth Client ID**
  * Choose type: **Web Application**
  * Add Authorized Redirect URI:

    ```
    http://localhost:8000/auth/callback
    ```

### 2. Download the `credentials.json` file

Save it in the project root.

---

## 🧾 Environment Variables

Create a `.env` file in the root directory:

```env
BASE_URL=http://localhost:8000
GOOGLE_CLIENT_SECRETS=credentials.json
GOOGLE_TOKEN_PATH=token.json

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_EMBED_FALLBACKS=bge-m3,mxbai-embed-large
```

---

## ▶️ Running the Backend

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will be live at:

```
http://localhost:8000
```

---

## 🧭 API Endpoints

| Endpoint         | Method | Description                                    |
| ---------------- | ------ | ---------------------------------------------- |
| `/`              | GET    | Health check                                   |
| `/auth/start`    | GET    | Start Google OAuth2 flow                       |
| `/auth/callback` | GET    | OAuth redirect handler                         |
| `/auth/fetch`    | GET    | Fetch Gmail & Drive data and index embeddings  |
| `/ask`           | POST   | Ask a question (JSON: `{ "question": "..." }`) |
| `/debug/chroma`  | GET    | Inspect stored ChromaDB documents              |

---

## 💡 Usage Flow

1. Start the backend

   ```bash
   uvicorn main:app --reload
   ```

2. Visit:

   ```
   http://localhost:8000/auth/start
   ```

   → Log in with your Google account
   → Grants access to Gmail + Drive

3. Fetch data:

   ```
   http://localhost:8000/auth/fetch
   ```

   This will:

   * Download your Gmail snippets & Drive file metadata
   * Generate embeddings using Ollama
   * Store them in ChromaDB

4. Ask a question:

   ```bash
   curl -X POST http://localhost:8000/ask \
   -H "Content-Type: application/json" \
   -d '{"question": "What documents did I recently upload?"}'
   ```

5. (Optional) Inspect your vector DB:

   ```
   http://localhost:8000/debug/chroma
   ```

---

## 🧹 Reset Data

To clear your ChromaDB and start fresh:

```bash
rm -rf chroma_db token.json
```

---

## 🧠 Notes

* This is for **local experimentation** only.
  For production, secure OAuth tokens and sensitive data properly.
* If embeddings fail, ensure Ollama models are available:

  ```bash
  ollama list
  ```
* Make sure your **frontend BASE_URL** matches the backend config (default: `localhost:3000`).

---

## 🧰 Tech Stack

| Component       | Purpose                      |
| --------------- | ---------------------------- |
| **FastAPI**     | API framework                |
| **Ollama**      | LLM inference and embeddings |
| **ChromaDB**    | Vector store                 |
| **Google APIs** | Gmail + Drive data access    |
| **dotenv**      | Environment config           |

---




