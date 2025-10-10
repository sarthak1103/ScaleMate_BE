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
# Hugging Face Config
# ----------------------------
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "gpt2")
HF_EMBED_FALLBACKS = [
    m.strip() for m in os.getenv("HF_EMBED_FALLBACKS", "BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5").split(",") if m.strip()
]
HF_CHAT_FALLBACKS = [
    m.strip() for m in os.getenv("HF_CHAT_FALLBACKS", "distilgpt2").split(",") if m.strip()
]

# ----------------------------
# Embeddings
# ----------------------------
def huggingface_embed(text: str):
    """Get embeddings from Hugging Face with fallbacks"""
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY environment variable is required")
    
    def call_embed(model, input_data):
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        resp = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": input_data},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    # Try main model
    try:
        data = call_embed(HF_EMBED_MODEL, text)
        if isinstance(data, list) and len(data) > 0:
            # BAAI models return the embedding vector directly as a list of floats
            return data
    except Exception:
        pass

    # Try fallbacks
    for model in HF_EMBED_FALLBACKS:
        try:
            data = call_embed(model, text)
            if isinstance(data, list) and len(data) > 0:
                # BAAI models return the embedding vector directly as a list of floats
                return data
        except Exception:
            continue

    raise RuntimeError("No embeddings available. Check your Hugging Face API key and model availability.")


def store_embeddings(texts):
    """Embed and store text into ChromaDB"""
    valid_ids, valid_docs, valid_embs = [], [], []
    for i, raw in enumerate(texts):
        text = (raw or "").strip()
        if not text or len(text) < 10:  # skip short texts
            continue
        try:
            emb = huggingface_embed(text)
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
    q_emb = huggingface_embed(query)

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

    # Simple response based on context (since many HF chat models are not available)
    # For now, we'll provide a basic response based on the retrieved context
    if "No relevant information found" in context or len(context.strip()) < 50:
        return "No relevant information found in your data. Please try a different question or make sure you've fetched your Gmail and Drive data first."
    
    # Extract key information from context
    context_lines = context.split('\n')[:3]  # Take first 3 lines
    relevant_info = '\n'.join([line.strip() for line in context_lines if line.strip()])
    
    return f"Based on your data, here's what I found:\n\n{relevant_info}\n\nThis information is related to your question: '{query}'. For more detailed analysis, you might want to ask more specific questions about your emails or files."


# ----------------------------
# API Routes
# ----------------------------
@app.get("/")
def health():
    return {"msg": "🚀 Hackathon MVP running with Hugging Face"}


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
