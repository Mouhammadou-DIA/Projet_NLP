"""
Index Conversations Script - Reddit RAG Chatbot
Create vector embeddings and index conversations in ChromaDB
"""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger, log_startup
from src.config.settings import settings
from src.core.embeddings import get_embedding_service
from src.core.vector_store import get_vector_store_service
from src.utils.data_loader import load_conversations


logger = get_logger(__name__)


def main():
    """Main indexing function"""
    log_startup()

    logger.info("=" * 60)
    logger.info("INDEXING - Reddit RAG Chatbot")
    logger.info("=" * 60)

    # Initialize services
    logger.info("\n Initializing services...")
    embedding_service = get_embedding_service()
    vector_store = get_vector_store_service()

    logger.info(f"  Embedding model: {embedding_service.model_name}")
    logger.info(f"  Embedding dimension: {embedding_service.embedding_dim}")
    logger.info(f"  Vector store: {vector_store.collection_name}")

    # Load conversations
    logger.info("\n Loading conversations...")
    conversations = load_conversations()
    logger.info(f"✓ Loaded {len(conversations)} conversations")

    # Check if already indexed
    existing_count = vector_store.count()
    if existing_count > 0:
        logger.warning(f" Vector store already contains {existing_count} documents")
        response = input("Reset and re-index? (yes/no): ")
        if response.lower() == "yes":
            vector_store.reset()
            logger.info("✓ Vector store reset")
        else:
            logger.info("Skipping indexing")
            return

    # Create embeddings
    logger.info("\n Creating embeddings...")
    logger.info(f"  Batch size: {settings.EMBEDDING_BATCH_SIZE}")
    logger.info("  This may take 10-15 minutes for 56k conversations...")

    texts = [conv.full_text for conv in conversations]
    embeddings = embedding_service.embed_batch(texts, show_progress=True)

    logger.info(f"✓ Created {len(embeddings)} embeddings")

    # Index in vector store
    logger.info("\n Indexing in ChromaDB...")

    batch_size = 1000
    total_indexed = 0

    for i in range(0, len(conversations), batch_size):
        batch_convs = conversations[i : i + batch_size]
        batch_embeds = embeddings[i : i + batch_size]

        success = vector_store.add_conversations(batch_convs, batch_embeds)

        if success:
            total_indexed += len(batch_convs)
            logger.info(f"  Indexed {total_indexed}/{len(conversations)} conversations")
        else:
            logger.error(f"Failed to index batch {i // batch_size + 1}")

    # Verify
    logger.info("\n Verification...")
    final_count = vector_store.count()
    logger.info(f"  Total documents in vector store: {final_count}")

    if final_count == len(conversations):
        logger.info(" All conversations indexed successfully")
    else:
        logger.warning(f"  Mismatch: {len(conversations)} loaded, {final_count} indexed")

    # Test search
    logger.info("\n Testing search...")
    test_query = "What phone should I buy?"
    logger.info(f"  Test query: '{test_query}'")

    test_embedding = embedding_service.embed_text(test_query)
    test_results = vector_store.search(test_embedding, n_results=3)

    logger.info(f"  Found {len(test_results)} results:")
    for i, result in enumerate(test_results, 1):
        logger.info(f"    {i}. Score: {result.score:.3f} - {result.conversation.context[:50]}...")

    logger.info("\n" + "=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nVector store ready at: {vector_store.persist_directory}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e!s}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
