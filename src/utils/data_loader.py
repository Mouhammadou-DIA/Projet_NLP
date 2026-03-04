"""
Data Loader - Professional Reddit RAG Chatbot
Load and parse conversation data from various formats
"""

import json
from pathlib import Path

import pandas as pd

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.models.schemas import Conversation


logger = get_logger(__name__)


class DataLoader:
    """
    Data loader for conversation datasets

    Supports:
    - CSV files
    - JSON files
    - Pandas DataFrames
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir or settings.PROCESSED_DATA_DIR
        logger.info(f"DataLoader initialized with directory: {self.data_dir}")

    def load_from_csv(self, filepath: Path, **kwargs) -> list[Conversation]:
        """
        Load conversations from CSV file

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            List of Conversation objects
        """
        try:
            logger.info(f"Loading conversations from CSV: {filepath}")

            df = pd.read_csv(filepath, **kwargs)
            conversations = self._dataframe_to_conversations(df)

            logger.info(f"✓ Loaded {len(conversations)} conversations from CSV")
            return conversations

        except Exception as e:
            logger.error(f"Failed to load CSV: {e!s}")
            raise

    def load_from_json(self, filepath: Path) -> list[Conversation]:
        """
        Load conversations from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            List of Conversation objects
        """
        try:
            logger.info(f"Loading conversations from JSON: {filepath}")

            with Path(filepath).open(encoding="utf-8") as f:
                data = json.load(f)

            conversations = [Conversation(**item) for item in data]

            logger.info(f"✓ Loaded {len(conversations)} conversations from JSON")
            return conversations

        except Exception as e:
            logger.error(f"Failed to load JSON: {e!s}")
            raise

    def save_to_json(self, conversations: list[Conversation], filepath: Path):
        """
        Save conversations to JSON file

        Args:
            conversations: List of Conversation objects
            filepath: Path to save file
        """
        try:
            logger.info(f"Saving {len(conversations)} conversations to JSON: {filepath}")

            data = [conv.dict() for conv in conversations]

            with Path(filepath).open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ Saved conversations to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save JSON: {e!s}")
            raise

    def _dataframe_to_conversations(self, df: pd.DataFrame) -> list[Conversation]:
        """
        Convert DataFrame to Conversation objects

        Args:
            df: DataFrame with columns: id, context, response, (optional: follow_up)

        Returns:
            List of Conversation objects
        """
        conversations = []

        # Detect column names (handle different formats)
        if "0" in df.columns and "1" in df.columns:
            # Format: 0, 1, 2 (context, response, follow_up)
            context_col = "0"
            response_col = "1"
            follow_up_col = "2" if "2" in df.columns else None
        else:
            # Standard format
            context_col = "context"
            response_col = "response"
            follow_up_col = "follow_up" if "follow_up" in df.columns else None

        for idx, row in df.iterrows():
            try:
                conv = Conversation(
                    id=int(idx) if "id" not in df.columns else int(row["id"]),
                    context=str(row[context_col]),
                    response=str(row[response_col]),
                    follow_up=str(row[follow_up_col])
                    if follow_up_col and pd.notna(row[follow_up_col])
                    else None,
                )
                conversations.append(conv)

            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e!s}")
                continue

        return conversations

    def get_stats(self, conversations: list[Conversation]) -> dict:
        """
        Get statistics about conversations

        Args:
            conversations: List of Conversation objects

        Returns:
            Statistics dictionary
        """
        if not conversations:
            return {"total": 0}

        contexts = [conv.context for conv in conversations]
        responses = [conv.response for conv in conversations]

        return {
            "total": len(conversations),
            "avg_context_length": sum(len(c) for c in contexts) / len(contexts),
            "avg_response_length": sum(len(r) for r in responses) / len(responses),
            "max_context_length": max(len(c) for c in contexts),
            "max_response_length": max(len(r) for r in responses),
            "with_follow_up": sum(1 for c in conversations if c.follow_up),
        }


def load_conversations(filepath: Path | None = None) -> list[Conversation]:
    """
    Convenience function to load conversations

    Args:
        filepath: Path to data file (CSV or JSON)

    Returns:
        List of Conversation objects
    """
    loader = DataLoader()

    if filepath is None:
        # Try to find data file in processed directory
        processed_dir = settings.PROCESSED_DATA_DIR

        # Try JSON first
        json_path = processed_dir / "conversations.json"
        if json_path.exists():
            return loader.load_from_json(json_path)

        # Try CSV
        csv_path = processed_dir / "conversations.csv"
        if csv_path.exists():
            return loader.load_from_csv(csv_path)

        raise FileNotFoundError("No conversation data found")

    # Load from specified file
    filepath = Path(filepath)

    if filepath.suffix == ".json":
        return loader.load_from_json(filepath)
    elif filepath.suffix == ".csv":
        return loader.load_from_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
