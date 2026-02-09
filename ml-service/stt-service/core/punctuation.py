"""Transformer-based punctuation restoration for STT output.

Supports multiple languages including Indic languages using
the deepmultilingualpunctuation library.

Dependencies:
    pip install deepmultilingualpunctuation
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from loguru import logger

# Try to import punctuation model
try:
    from deepmultilingualpunctuation import PunctuationModel
    PUNCTUATION_AVAILABLE = True
except ImportError:
    PUNCTUATION_AVAILABLE = False
    logger.warning(
        "deepmultilingualpunctuation not available. "
        "Install with: pip install deepmultilingualpunctuation"
    )

# Cached punctuation model
_punctuation_model: Optional["PunctuationModel"] = None


# Language code mappings
LANG_MAP = {
    "hi-IN": "hi",
    "ta-IN": "ta",
    "te-IN": "te",
    "bn-IN": "bn",
    "mr-IN": "mr",
    "gu-IN": "gu",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "en-IN": "en",
    "en-US": "en",
    "en-GB": "en",
}

# Supported languages by the model
SUPPORTED_LANGUAGES = {
    "en", "de", "fr", "es", "it", "nl", "pt", "pl", "cs", "sk",
    "sl", "hr", "bg", "ro", "hu", "fi", "et", "lv", "lt", "sv",
    "da", "no", "is", "el", "tr", "ru", "uk", "be", "sr", "mk",
    "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or",
}


def get_punctuation_model() -> Optional["PunctuationModel"]:
    """Get cached punctuation model.

    Returns:
        PunctuationModel instance or None if not available
    """
    global _punctuation_model

    if not PUNCTUATION_AVAILABLE:
        return None

    if _punctuation_model is None:
        logger.info("Loading punctuation model...")
        try:
            _punctuation_model = PunctuationModel()
            logger.info("Punctuation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load punctuation model: {e}")
            return None

    return _punctuation_model


def add_punctuation_transformer(text: str, language: str = "en") -> str:
    """Add punctuation using transformer model.

    Args:
        text: Input text without punctuation
        language: Language code

    Returns:
        Text with punctuation added
    """
    model = get_punctuation_model()
    if model is None:
        return add_punctuation_rules(text, language)

    lang = LANG_MAP.get(language, language)

    # Check if language is supported
    if lang not in SUPPORTED_LANGUAGES:
        logger.debug(f"Language {lang} not supported for punctuation, using rules")
        return add_punctuation_rules(text, language)

    try:
        # The model expects clean text
        cleaned_text = clean_text_for_punctuation(text)
        if not cleaned_text.strip():
            return text

        # Restore punctuation
        punctuated = model.restore_punctuation(cleaned_text)
        return punctuated

    except Exception as e:
        logger.error(f"Punctuation restoration failed: {e}")
        return add_punctuation_rules(text, language)


def clean_text_for_punctuation(text: str) -> str:
    """Clean text before punctuation restoration.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove existing punctuation (the model will add its own)
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def add_punctuation_rules(text: str, language: str = "en") -> str:
    """Add basic punctuation using rules.

    Fallback when transformer model is not available.

    Args:
        text: Input text
        language: Language code

    Returns:
        Text with basic punctuation
    """
    if not text or not text.strip():
        return text

    # Language-specific sentence endings
    lang = LANG_MAP.get(language, language)

    # Question indicators by language
    question_words: Dict[str, List[str]] = {
        "en": ["what", "where", "when", "why", "how", "who", "which", "whose", "whom", "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should"],
        "hi": ["क्या", "कहाँ", "कब", "क्यों", "कैसे", "कौन", "किसका", "किसको", "किधर"],
        "ta": ["என்ன", "எங்கே", "எப்போது", "ஏன்", "எப்படி", "யார்"],
        "te": ["ఏమిటి", "ఎక్కడ", "ఎప్పుడు", "ఎందుకు", "ఎలా", "ఎవరు"],
        "bn": ["কি", "কোথায়", "কখন", "কেন", "কিভাবে", "কে"],
    }

    # Get question words for the language
    q_words = question_words.get(lang, question_words["en"])

    # Split into sentences (rough)
    sentences = re.split(r'(?<=[.!?।॥])\s+', text)

    if len(sentences) == 1 and not any(text.strip().endswith(p) for p in '.!?।॥'):
        # Single sentence without ending punctuation
        sentence = text.strip()

        # Check if it's a question
        first_word = sentence.split()[0].lower() if sentence.split() else ""
        last_word = sentence.split()[-1].lower() if sentence.split() else ""

        if first_word in q_words or last_word in q_words:
            return sentence + "?"
        else:
            # Use language-appropriate period
            if lang == "hi":
                return sentence + "।"
            elif lang in ["ta", "te", "kn", "ml"]:
                return sentence + "."
            else:
                return sentence + "."

    # Multiple sentences - ensure each ends with punctuation
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if already has ending punctuation
        if sentence and sentence[-1] in '.!?।॥':
            result.append(sentence)
        else:
            # Add appropriate punctuation
            first_word = sentence.split()[0].lower() if sentence.split() else ""
            if first_word in q_words:
                result.append(sentence + "?")
            elif lang == "hi":
                result.append(sentence + "।")
            else:
                result.append(sentence + ".")

    return " ".join(result)


def add_punctuation(text: str, language: str) -> str:
    """Add punctuation to text.

    Uses transformer model if available, falls back to rules.

    Args:
        text: Input text without (or with partial) punctuation
        language: Target language code (e.g., 'hi-IN', 'en-US')

    Returns:
        Text with punctuation restored
    """
    logger.debug(f"Adding punctuation for language {language}")

    if not text or not text.strip():
        return text

    # Try transformer-based punctuation first
    if PUNCTUATION_AVAILABLE:
        result = add_punctuation_transformer(text, language)
    else:
        result = add_punctuation_rules(text, language)

    logger.debug(f"Punctuated text: {result[:100]}..." if len(result) > 100 else f"Punctuated text: {result}")

    return result


def restore_capitalization(text: str, language: str = "en") -> str:
    """Restore proper capitalization to text.

    Args:
        text: Input text
        language: Language code

    Returns:
        Text with proper capitalization
    """
    if not text:
        return text

    # For non-Latin scripts, skip capitalization
    lang = LANG_MAP.get(language, language)
    if lang in ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or"]:
        return text

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # Capitalize after sentence endings
    text = re.sub(
        r'([.!?]\s+)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        text
    )

    return text


def format_transcript(
    text: str,
    language: str,
    add_punct: bool = True,
    fix_caps: bool = True,
) -> str:
    """Format transcript with punctuation and capitalization.

    Args:
        text: Raw transcript text
        language: Language code
        add_punct: Whether to add punctuation
        fix_caps: Whether to fix capitalization

    Returns:
        Formatted transcript
    """
    result = text

    if add_punct:
        result = add_punctuation(result, language)

    if fix_caps:
        result = restore_capitalization(result, language)

    return result
