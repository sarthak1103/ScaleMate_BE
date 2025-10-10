# Hugging Face Setup Guide

## 🚀 Quick Setup

### 1. Get Hugging Face API Key
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions
5. Copy the token

### 2. Set Environment Variables
Create a `.env` file in your project root:

```bash
# Google OAuth Configuration
GOOGLE_CLIENT_SECRETS=credentials.json
GOOGLE_TOKEN_PATH=token.json
BASE_URL=http://localhost:8000

# Hugging Face Configuration
HUGGINGFACE_API_KEY=your_actual_api_key_here
HF_EMBED_MODEL=BAAI/bge-small-en-v1.5
HF_CHAT_MODEL=gpt2
HF_EMBED_FALLBACKS=BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5
HF_CHAT_FALLBACKS=distilgpt2
```

### 3. Install Dependencies
```bash
pip install requests fastapi uvicorn python-dotenv google-auth-oauthlib google-api-python-client chromadb
```

### 4. Run the Application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Free Tier Limits
- **50 requests/hour** for free tier
- Perfect for development and small-scale testing
- Consider upgrading for production use

## 🔧 Models Used
- **Embeddings**: `BAAI/bge-small-en-v1.5` (384 dimensions) - ✅ Working
- **Chat**: Simple context-based responses (many HF chat models not available)
- **Fallbacks**: BAAI embedding models for reliability

## 🚨 Important Notes
- The API key is required for all Hugging Face API calls
- Models may take a few seconds to load on first use
- Free tier has rate limits - implement caching for production
