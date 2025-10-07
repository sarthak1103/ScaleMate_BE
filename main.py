import os
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Google imports
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest

# Vector DB
import chromadb

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI()

# Allow React frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Config
# ----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS", "credentials.json")

# Vector DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("user_data")

# Store OAuth flows
oauth_flows_by_state = {}

# Allow insecure transport for local dev
if BASE_URL.startswith(("http://localhost", "http://127.0.0.1")):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ----------------------------
# Google OAuth Helpers
# ----------------------------
def save_credentials(creds: Credentials):
    """Save Google OAuth credentials to disk"""
    data = {
        "token": creds.token,
        "refresh_token": getattr(creds, "refresh_token", None),
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    with open(TOKEN_PATH, "w") as f:
        json.dump(data, f)


def load_credentials() -> Credentials | None:
    """Load credentials if available"""
    if os.path.exists(TOKEN_PATH):
        try:
            return Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception:
            return None
    return None


def ensure_valid_credentials() -> Credentials | None:
    """Ensure token is fresh, refresh if expired"""
    creds = load_credentials()
    if not creds:
        return None
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleRequest())
            save_credentials(creds)
        except Exception:
            return None
    return creds


# ----------------------------
# Google APIs
# ----------------------------
def get_gmail_data(creds, max_results=5):
    """Fetch latest Gmail messages"""
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])
    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId="me", id=msg["id"]).execute()
        emails.append(txt["snippet"])
    return emails


def get_drive_data(creds, max_results=5):
    """Fetch latest Google Drive files"""
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    results = service.files().list(pageSize=max_results, fields="files(id, name, mimeType)").execute()
    files = results.get("files", [])
    return [f"{f['name']} ({f['mimeType']})" for f in files]


# ----------------------------
# Ollama Config
# ----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_EMBED_FALLBACKS = [
    m.strip() for m in os.getenv("OLLAMA_EMBED_FALLBACKS", "bge-m3,mxbai-embed-large").split(",") if m.strip()
]

# ----------------------------
# Embeddings
# ----------------------------
def ollama_embed(text: str):
    """Get embeddings from Ollama with fallbacks"""
    def call_embed(model, input_data):
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": model, "prompt": input_data},  # Ollama uses 'prompt'
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    # Try main model
    try:
        data = call_embed(OLLAMA_EMBED_MODEL, text)
        if isinstance(data.get("embedding"), list) and data["embedding"]:
            return data["embedding"]
    except Exception:
        pass

    # Try fallbacks
    for model in OLLAMA_EMBED_FALLBACKS:
        try:
            data = call_embed(model, text)
            if isinstance(data.get("embedding"), list) and data["embedding"]:
                return data["embedding"]
        except Exception:
            continue

    raise RuntimeError("No embeddings available. Make sure models are pulled via `ollama pull <model>`")


def store_embeddings(texts):
    """Embed and store text into ChromaDB"""
    valid_ids, valid_docs, valid_embs = [], [], []
    for i, raw in enumerate(texts):
        text = (raw or "").strip()
        if not text or len(text) < 10:  # skip short texts
            continue
        try:
            emb = ollama_embed(text)
            valid_ids.append(f"id-{i}")
            valid_docs.append(text)
            valid_embs.append(emb)
        except Exception:
            continue

    if valid_ids:
        collection.add(ids=valid_ids, documents=valid_docs, embeddings=valid_embs)


# ----------------------------
# RAG Search
# ----------------------------
def rag_search(query):
    """Perform Retrieval-Augmented Generation over user data"""
    q_emb = ollama_embed(query)

    # Check collection size
    try:
        total_items = collection.count()
    except Exception:
        total_items = 0
    if total_items <= 0:
        return "I don't have any indexed data yet. Please hit /auth/fetch first."

    # Query Chroma
    k = min(10, total_items)
    results = collection.query(query_embeddings=[q_emb], n_results=k)

    # Flatten docs
    documents = results.get("documents") or []
    flat_docs = [d for sub in documents for d in (sub if isinstance(sub, list) else [sub]) if d]
    if not flat_docs:
        return "I don't have any indexed data yet. Please hit /auth/fetch first."

    # Build context
    top_docs = flat_docs[:10]
    context = "\n".join(top_docs)

    # Chat with Ollama
    prompt = f"""You are a helpful assistant answering QUESTIONS ABOUT THE USER'S DATA ONLY.
Use ONLY the information in Context. 
If no relevant info: reply with "No relevant information found in context."

Context:
{context}

Question: {query}
"""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return (response.json().get("response") or "").strip()


# ----------------------------
# API Routes
# ----------------------------
@app.get("/")
def health():
    return {"msg": "🚀 Hackathon MVP running with Ollama"}


@app.get("/auth/start")
def auth_start():
    """Begin OAuth flow"""
    redirect_uri = f"{BASE_URL}/auth/callback"
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=redirect_uri)
    authorization_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    oauth_flows_by_state[state] = flow
    return RedirectResponse(authorization_url)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle OAuth callback"""
    params = dict(request.query_params)
    state = params.get("state")

    if state and state in oauth_flows_by_state:
        flow = oauth_flows_by_state.pop(state)
        try:
            flow.fetch_token(authorization_response=str(request.url))
            creds = flow.credentials
            save_credentials(creds)
            frontend_url = f"{BASE_URL.replace(':8000', ':3000')}/auth/callback?auth=completed"
            return RedirectResponse(url=frontend_url)
        except Exception as e:
            frontend_url = f"{BASE_URL.replace(':8000', ':3000')}/auth/callback?auth=error&error={str(e)}"
            return RedirectResponse(url=frontend_url)

    return RedirectResponse(f"{BASE_URL.replace(':8000', ':3000')}/auth/callback?auth=error&error=Invalid OAuth state")


@app.get("/auth/fetch")
def auth_fetch():
    """Fetch Gmail + Drive and store embeddings"""
    creds = ensure_valid_credentials()
    if not creds:
        return {"error": "Not authenticated. Visit /auth/start to login."}

    try:
        emails = get_gmail_data(creds)
        files = get_drive_data(creds)
        combined = emails + files
        store_embeddings(combined)
        return {"emails": emails, "files": files}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
async def ask(request: Request):
    """Answer user queries based on indexed data"""
    body = await request.json()
    query = body.get("question")
    if not query:
        return {"error": "Question is required"}
    try:
        answer = rag_search(query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/chroma")
def debug_chroma():
    """Inspect vector DB content"""
    try:
        count = collection.count()
        sample = collection.get(limit=5)
        return {
            "total_documents": count,
            "sample_ids": sample.get("ids", []),
            "sample_documents": sample.get("documents", []),
            "sample_metadatas": sample.get("metadatas", []),
        }
    except Exception as e:
        return {"error": str(e)}
