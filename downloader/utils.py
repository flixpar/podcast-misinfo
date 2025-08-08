"""
Utility functions for string normalization and ID generation.
"""

import re
import unicodedata
from typing import Optional


def normalize_id(value: Optional[str], max_length: int = 100) -> str:
    """
    Create a normalized identifier from a string.

    Rules:
    - Lowercase
    - Strip accents/diacritics (ASCII only)
    - Replace any sequence of non-alphanumeric characters with single '-'
    - Remove leading/trailing '-'
    - Enforce max_length
    - Fallback to 'unknown' if empty
    """
    if not value:
        return "unknown"

    # Normalize to NFKD and drop diacritics, keep ASCII
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    value = value.lower()

    # Replace any non-alphanumeric with '-' and collapse repeats
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")

    # Enforce max length (avoid trimming to empty)
    if max_length > 0 and len(value) > max_length:
        value = value[:max_length].rstrip("-")

    return value or "unknown"

