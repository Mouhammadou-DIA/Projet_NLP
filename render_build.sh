#!/usr/bin/env bash
# ============================================================
# Render Build Script - Reddit RAG Chatbot
# Runs during Render's build phase
# ============================================================
set -o errexit

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Pre-downloading ML models ==="
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
print('Downloading embedding model...')
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('Downloading cross-encoder model...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Models downloaded successfully!')
"

echo "=== Indexing conversations into ChromaDB ==="
python -c "
import sys
sys.path.insert(0, '.')

from src.core.embeddings import get_embedding_service
from src.core.vector_store import get_vector_store_service
from src.utils.data_loader import load_conversations

print('Loading conversations...')
conversations = load_conversations()
print(f'Loaded {len(conversations)} conversations')

print('Initializing services...')
embedding_service = get_embedding_service()
vector_store = get_vector_store_service()

existing = vector_store.count()
if existing > 0:
    print(f'Vector store already has {existing} documents, skipping indexing')
else:
    print('Creating embeddings (this may take a few minutes)...')
    texts = [conv.full_text for conv in conversations]
    embeddings = embedding_service.embed_batch(texts, show_progress=True)

    print('Indexing in ChromaDB...')
    batch_size = 1000
    for i in range(0, len(conversations), batch_size):
        batch_convs = conversations[i:i+batch_size]
        batch_embeds = embeddings[i:i+batch_size]
        vector_store.add_conversations(batch_convs, batch_embeds)
        print(f'  Indexed {min(i+batch_size, len(conversations))}/{len(conversations)}')

    print(f'Indexing complete! Total: {vector_store.count()} documents')
"

echo "=== Build complete ==="
