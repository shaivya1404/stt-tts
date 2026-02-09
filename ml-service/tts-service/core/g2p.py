"""Grapheme-to-Phoneme (G2P) conversion for Indic languages.

Supports:
- Devanagari (Hindi, Marathi, Sanskrit)
- Tamil
- Telugu
- Bengali
- Kannada
- Malayalam
- Gujarati

Uses aksharamukha for transliteration and IPA conversion.

Dependencies:
    pip install aksharamukha
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Try to import aksharamukha
try:
    from aksharamukha import transliterate
    AKSHARAMUKHA_AVAILABLE = True
except ImportError:
    AKSHARAMUKHA_AVAILABLE = False
    logger.warning("aksharamukha not available. Install with: pip install aksharamukha")


# Script detection patterns
SCRIPT_PATTERNS = {
    "Devanagari": re.compile(r'[\u0900-\u097F]'),  # Hindi, Marathi, Sanskrit
    "Tamil": re.compile(r'[\u0B80-\u0BFF]'),
    "Telugu": re.compile(r'[\u0C00-\u0C7F]'),
    "Bengali": re.compile(r'[\u0980-\u09FF]'),
    "Kannada": re.compile(r'[\u0C80-\u0CFF]'),
    "Malayalam": re.compile(r'[\u0D00-\u0D7F]'),
    "Gujarati": re.compile(r'[\u0A80-\u0AFF]'),
    "Gurmukhi": re.compile(r'[\u0A00-\u0A7F]'),  # Punjabi
    "Oriya": re.compile(r'[\u0B00-\u0B7F]'),
    "Latin": re.compile(r'[A-Za-z]'),
}

# Language to script mapping
LANG_TO_SCRIPT = {
    "hi": "Devanagari",
    "hi-IN": "Devanagari",
    "mr": "Devanagari",
    "mr-IN": "Devanagari",
    "ta": "Tamil",
    "ta-IN": "Tamil",
    "te": "Telugu",
    "te-IN": "Telugu",
    "bn": "Bengali",
    "bn-IN": "Bengali",
    "kn": "Kannada",
    "kn-IN": "Kannada",
    "ml": "Malayalam",
    "ml-IN": "Malayalam",
    "gu": "Gujarati",
    "gu-IN": "Gujarati",
    "pa": "Gurmukhi",
    "pa-IN": "Gurmukhi",
    "or": "Oriya",
    "or-IN": "Oriya",
    "en": "Latin",
    "en-IN": "Latin",
}

# Hindi schwa deletion rules (simplified)
# In Hindi, schwa (inherent 'a' vowel) is often deleted at word ends and between consonants
SCHWA_DELETION_PATTERNS = [
    # Word-final schwa deletion
    (r'(\S)a$', r'\1'),
    # Schwa deletion before consonant clusters
    (r'a([kgcjṭḍtnpbmyrlvśṣsh])([aāiīuūeēoō])', r'\1\2'),
]


def detect_script(text: str) -> str:
    """Detect the script of the given text.

    Args:
        text: Input text

    Returns:
        Detected script name (e.g., 'Devanagari', 'Tamil')
    """
    script_counts = {}

    for script, pattern in SCRIPT_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            script_counts[script] = len(matches)

    if not script_counts:
        return "Latin"  # Default

    return max(script_counts, key=script_counts.get)


def transliterate_to_ipa(text: str, source_script: str) -> str:
    """Transliterate text to IPA using aksharamukha.

    Args:
        text: Input text in Indic script
        source_script: Source script name

    Returns:
        IPA transcription
    """
    if not AKSHARAMUKHA_AVAILABLE:
        return text

    try:
        # Transliterate to IPA
        ipa = transliterate.process(source_script, 'IPA', text)
        return ipa
    except Exception as e:
        logger.warning(f"IPA transliteration failed: {e}")
        return text


def transliterate_to_roman(text: str, source_script: str) -> str:
    """Transliterate text to romanized form.

    Args:
        text: Input text in Indic script
        source_script: Source script name

    Returns:
        Romanized text (ISO 15919 or IAST)
    """
    if not AKSHARAMUKHA_AVAILABLE:
        return text

    try:
        # Use ISO 15919 for accurate romanization
        roman = transliterate.process(source_script, 'ISO', text)
        return roman
    except Exception as e:
        logger.warning(f"Romanization failed: {e}")
        return text


def apply_schwa_deletion(text: str, language: str) -> str:
    """Apply schwa deletion rules for Hindi.

    In Hindi, the inherent 'a' vowel (schwa) is often not pronounced
    at word ends and in certain consonant clusters.

    Args:
        text: Romanized text
        language: Language code

    Returns:
        Text with schwa deletion applied
    """
    if language not in ["hi", "hi-IN"]:
        return text

    result = text
    for pattern, replacement in SCHWA_DELETION_PATTERNS:
        result = re.sub(pattern, replacement, result)

    return result


def text_to_phonemes_aksharamukha(text: str, language: str) -> List[str]:
    """Convert text to phonemes using aksharamukha.

    Args:
        text: Input text
        language: Language code

    Returns:
        List of phoneme symbols
    """
    script = LANG_TO_SCRIPT.get(language, detect_script(text))

    # Get IPA transcription
    ipa = transliterate_to_ipa(text, script)

    # Apply schwa deletion for Hindi
    ipa = apply_schwa_deletion(ipa, language)

    # Split into individual phonemes
    # IPA uses combining characters, so we need to handle them
    phonemes = []
    current_phoneme = ""

    for char in ipa:
        # Combining characters (diacritics)
        if '\u0300' <= char <= '\u036f' or '\u1dc0' <= char <= '\u1dff':
            current_phoneme += char
        # Modifier letters
        elif char in 'ˈˌːˑ':
            current_phoneme += char
        else:
            if current_phoneme:
                phonemes.append(current_phoneme)
            current_phoneme = char

    if current_phoneme:
        phonemes.append(current_phoneme)

    # Filter out spaces and empty strings
    phonemes = [p for p in phonemes if p.strip()]

    return phonemes


def text_to_phonemes_simple(text: str, language: str) -> List[str]:
    """Simple phoneme conversion without aksharamukha.

    Falls back to romanization-based approach.

    Args:
        text: Input text
        language: Language code

    Returns:
        List of pseudo-phonemes
    """
    # For Latin script, just return words
    script = LANG_TO_SCRIPT.get(language, detect_script(text))
    if script == "Latin":
        return text.lower().split()

    # For Indic scripts, use character-level tokenization
    phonemes = []
    for char in text:
        if char.strip() and not char.isspace():
            phonemes.append(char)

    return phonemes


def text_to_phonemes(normalized_text: str, language: str) -> List[str]:
    """Convert normalized text to phoneme sequence.

    Args:
        normalized_text: Pre-normalized text
        language: Target language code

    Returns:
        List of phoneme symbols (IPA or pseudo-phonemes)
    """
    logger.debug(f"Converting text to phonemes for language {language}")

    if not normalized_text or not normalized_text.strip():
        return []

    # Use aksharamukha if available
    if AKSHARAMUKHA_AVAILABLE:
        phonemes = text_to_phonemes_aksharamukha(normalized_text, language)
    else:
        phonemes = text_to_phonemes_simple(normalized_text, language)

    logger.debug(f"Generated {len(phonemes)} phonemes")
    return phonemes


def phonemes_to_ids(phonemes: List[str], vocab: Dict[str, int]) -> List[int]:
    """Convert phonemes to vocabulary IDs.

    Args:
        phonemes: List of phoneme symbols
        vocab: Phoneme to ID mapping

    Returns:
        List of phoneme IDs
    """
    unk_id = vocab.get("<unk>", 0)
    return [vocab.get(p, unk_id) for p in phonemes]


def get_phoneme_vocabulary(language: str) -> Dict[str, int]:
    """Get phoneme vocabulary for a language.

    Args:
        language: Language code

    Returns:
        Phoneme to ID mapping
    """
    # Base IPA phonemes common across Indic languages
    base_phonemes = [
        "<pad>", "<unk>", "<sos>", "<eos>",
        # Vowels
        "a", "ā", "i", "ī", "u", "ū", "e", "ē", "o", "ō",
        "ai", "au", "ṛ", "ṝ", "ḷ",
        # Consonants - Velars
        "k", "kh", "g", "gh", "ṅ",
        # Consonants - Palatals
        "c", "ch", "j", "jh", "ñ",
        # Consonants - Retroflexes
        "ṭ", "ṭh", "ḍ", "ḍh", "ṇ",
        # Consonants - Dentals
        "t", "th", "d", "dh", "n",
        # Consonants - Labials
        "p", "ph", "b", "bh", "m",
        # Consonants - Semi-vowels and sibilants
        "y", "r", "l", "v", "ś", "ṣ", "s", "h",
        # Nasalization and visarga
        "ṃ", "ḥ",
        # Spaces and punctuation
        " ", ".", ",", "?", "!",
    ]

    vocab = {p: i for i, p in enumerate(base_phonemes)}
    return vocab


def get_g2p_for_language(language: str):
    """Get a G2P converter function for the specified language.

    Args:
        language: Language code

    Returns:
        G2P function that takes text and returns phonemes
    """
    def g2p(text: str) -> List[str]:
        return text_to_phonemes(text, language)

    return g2p


# Pre-built G2P functions for common languages
g2p_hindi = get_g2p_for_language("hi-IN")
g2p_tamil = get_g2p_for_language("ta-IN")
g2p_telugu = get_g2p_for_language("te-IN")
g2p_bengali = get_g2p_for_language("bn-IN")
g2p_english = get_g2p_for_language("en-IN")
