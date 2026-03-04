"""
Data Preparation Script - Reddit RAG Chatbot
Prepare and clean conversation data for indexing
"""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger, log_startup
from src.config.settings import settings
from src.utils.data_loader import DataLoader
from src.utils.text_processor import TextProcessor


logger = get_logger(__name__)


def main():
    """Main data preparation function"""
    log_startup()

    logger.info("=" * 60)
    logger.info("DATA PREPARATION - Reddit RAG Chatbot")
    logger.info("=" * 60)

    # Paths
    raw_file = settings.RAW_DATA_DIR / "casual_data_windows.csv"
    output_file = settings.PROCESSED_DATA_DIR / "conversations.json"

    logger.info(f"Input: {raw_file}")
    logger.info(f"Output: {output_file}")

    # Initialize services
    loader = DataLoader()
    processor = TextProcessor()

    # Load data
    logger.info("\n Loading raw data...")
    conversations = loader.load_from_csv(raw_file)
    logger.info(f"✓ Loaded {len(conversations)} conversations")

    # Clean data
    logger.info("\n Cleaning conversations...")
    cleaned_conversations = []

    for conv in conversations:
        try:
            # Clean context and response
            conv.context = processor.clean_text(conv.context)
            conv.response = processor.clean_text(conv.response)

            # Validate
            if processor.is_valid_text(conv.context) and processor.is_valid_text(conv.response):
                cleaned_conversations.append(conv)
            else:
                logger.debug(f"Skipped invalid conversation {conv.id}")

        except Exception as e:
            logger.warning(f"Error cleaning conversation {conv.id}: {e!s}")
            continue

    logger.info(f"✓ Cleaned {len(cleaned_conversations)}/{len(conversations)} conversations")

    # Get statistics
    logger.info("\n Statistics:")
    stats = loader.get_stats(cleaned_conversations)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # Save processed data
    logger.info("\n Saving processed data...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    loader.save_to_json(cleaned_conversations, output_file)
    logger.info(f"✓ Saved to {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info(" DATA PREPARATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Data preparation failed: {e!s}")
        sys.exit(1)
