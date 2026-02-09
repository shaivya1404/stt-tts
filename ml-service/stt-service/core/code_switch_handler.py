"""Code-switching handler for Hindi-English mixed speech.

Provides:
- Romanized Hindi to Devanagari conversion
- Hindi-English code-switching detection
- Mixed language transcript normalization

Dependencies:
    pip install indic-transliteration
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Try to import transliteration library
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
    INDIC_TRANS_AVAILABLE = True
except ImportError:
    INDIC_TRANS_AVAILABLE = False
    logger.warning("indic-transliteration not available. Install with: pip install indic-transliteration")


# Common Hindi words in Roman script (Hinglish)
COMMON_HINDI_WORDS = {
    # Pronouns
    "main", "mein", "hum", "tum", "aap", "yeh", "woh", "wo", "kya", "kaun",
    "kab", "kahan", "kaise", "kyun", "kyon", "kitna", "kitni", "kitne",
    # Verbs
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge",
    "kar", "karo", "karna", "kiya", "karenge", "karunga",
    "ja", "jao", "jana", "gaya", "gayi", "jayenge", "jaunga",
    "aa", "aao", "aana", "aaya", "aayi", "aayenge",
    "de", "do", "dena", "diya", "denge", "dunga",
    "le", "lo", "lena", "liya", "lenge", "lunga",
    "bol", "bolo", "bolna", "bola", "bolenge",
    "dekh", "dekho", "dekhna", "dekha", "dekhenge",
    "sun", "suno", "sunna", "suna", "sunenge",
    "samajh", "samjho", "samajhna", "samjha", "samjhe",
    # Common words
    "acha", "accha", "achha", "theek", "thik", "sahi", "galat",
    "haan", "han", "nahi", "nahin", "nhi", "mat",
    "abhi", "aaj", "kal", "parso", "subah", "shaam", "raat",
    "ghar", "kaam", "paisa", "paise", "log", "baat",
    "bahut", "bohot", "thoda", "zyada", "kam", "jyada",
    "bhai", "behen", "dost", "yaar", "papa", "mummy",
    "chalo", "chal", "ruk", "ruko", "dekho", "suno",
    "karo", "karna", "milna", "milte", "milenge",
    # Conjunctions and particles
    "aur", "ya", "lekin", "par", "magar", "toh", "to", "bhi",
    "ke", "ki", "ka", "ko", "se", "mein", "par", "tak",
    "wala", "wali", "wale", "waala", "waali", "waale",
}

# Roman to Devanagari mapping (ITRANS-style)
ROMAN_TO_DEVANAGARI = {
    # Vowels
    "a": "अ", "aa": "आ", "i": "इ", "ii": "ई", "ee": "ई",
    "u": "उ", "uu": "ऊ", "oo": "ऊ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ",
    # Consonants
    "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ng": "ङ",
    "ch": "च", "chh": "छ", "j": "ज", "jh": "झ",
    "t": "त", "th": "थ", "d": "द", "dh": "ध", "n": "न",
    "T": "ट", "Th": "ठ", "D": "ड", "Dh": "ढ", "N": "ण",
    "p": "प", "ph": "फ", "f": "फ", "b": "ब", "bh": "भ", "m": "म",
    "y": "य", "r": "र", "l": "ल", "v": "व", "w": "व",
    "sh": "श", "Sh": "ष", "s": "स", "h": "ह",
    # Special
    "x": "क्ष", "gy": "ज्ञ", "gn": "ज्ञ",
}

# Matra (vowel signs) for consonants
DEVANAGARI_MATRAS = {
    "a": "", "aa": "ा", "i": "ि", "ii": "ी", "ee": "ी",
    "u": "ु", "uu": "ू", "oo": "ू",
    "e": "े", "ai": "ै", "o": "ो", "au": "ौ",
}


def is_hindi_word(word: str) -> bool:
    """Check if a word is likely Hindi (in Roman script).

    Args:
        word: Word to check

    Returns:
        True if word is likely Hindi
    """
    word_lower = word.lower().strip()
    return word_lower in COMMON_HINDI_WORDS


def is_english_word(word: str) -> bool:
    """Check if a word is likely English.

    Simple heuristic: contains only ASCII letters and is not a known Hindi word.

    Args:
        word: Word to check

    Returns:
        True if word is likely English
    """
    word_lower = word.lower().strip()

    # Check if it's ASCII-only
    if not word_lower.isascii() or not word_lower.isalpha():
        return False

    # Check if it's not a known Hindi word
    return word_lower not in COMMON_HINDI_WORDS


def is_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script.

    Args:
        text: Text to check

    Returns:
        True if text contains Devanagari
    """
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(devanagari_pattern.search(text))


def romanized_to_devanagari(text: str) -> str:
    """Convert Romanized Hindi (Hinglish) to Devanagari.

    Args:
        text: Romanized Hindi text

    Returns:
        Devanagari text
    """
    if INDIC_TRANS_AVAILABLE:
        try:
            # Use ITRANS to Devanagari conversion
            return indic_transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        except Exception as e:
            logger.warning(f"Transliteration failed: {e}")

    # Fallback: simple character-by-character conversion
    return _simple_roman_to_devanagari(text)


def _simple_roman_to_devanagari(text: str) -> str:
    """Simple Roman to Devanagari conversion.

    Args:
        text: Romanized text

    Returns:
        Devanagari text
    """
    result = []
    i = 0
    text_lower = text.lower()

    while i < len(text_lower):
        matched = False

        # Try longer sequences first
        for length in [3, 2, 1]:
            if i + length <= len(text_lower):
                seq = text_lower[i:i + length]
                if seq in ROMAN_TO_DEVANAGARI:
                    result.append(ROMAN_TO_DEVANAGARI[seq])
                    i += length
                    matched = True
                    break

        if not matched:
            result.append(text[i])
            i += 1

    return "".join(result)


def detect_code_switching(text: str) -> List[Tuple[str, str]]:
    """Detect code-switching in text.

    Args:
        text: Input text

    Returns:
        List of (segment, language) tuples
    """
    segments = []
    words = text.split()
    current_lang = None
    current_segment = []

    for word in words:
        # Detect language of word
        if is_devanagari(word):
            lang = "hi"
        elif is_hindi_word(word):
            lang = "hi-roman"
        elif is_english_word(word):
            lang = "en"
        else:
            lang = "unknown"

        # Check for language switch
        if current_lang is None:
            current_lang = lang
        elif lang != current_lang and lang != "unknown":
            if current_segment:
                segments.append((" ".join(current_segment), current_lang))
            current_segment = []
            current_lang = lang

        current_segment.append(word)

    # Add final segment
    if current_segment:
        segments.append((" ".join(current_segment), current_lang))

    return segments


def normalize_code_switched_text(
    text: str,
    target_script: str = "devanagari",
    preserve_english: bool = True,
) -> str:
    """Normalize code-switched text.

    Args:
        text: Input text with code-switching
        target_script: Target script ('devanagari' or 'roman')
        preserve_english: Whether to preserve English words as-is

    Returns:
        Normalized text
    """
    segments = detect_code_switching(text)
    result = []

    for segment, lang in segments:
        if lang == "en" and preserve_english:
            result.append(segment)
        elif lang == "hi-roman" and target_script == "devanagari":
            result.append(romanized_to_devanagari(segment))
        elif lang == "hi" and target_script == "roman":
            if INDIC_TRANS_AVAILABLE:
                try:
                    result.append(indic_transliterate(segment, sanscript.DEVANAGARI, sanscript.ITRANS))
                except Exception:
                    result.append(segment)
            else:
                result.append(segment)
        else:
            result.append(segment)

    return " ".join(result)


def handle_code_switching(
    transcript: str,
    source_language: str = "hi-IN",
    normalize_to_native: bool = True,
) -> Dict[str, any]:
    """Handle code-switching in transcript.

    Args:
        transcript: Raw transcript text
        source_language: Primary language
        normalize_to_native: Whether to normalize Romanized to native script

    Returns:
        Dict with normalized text and metadata
    """
    # Detect segments
    segments = detect_code_switching(transcript)

    # Calculate code-switching statistics
    total_words = len(transcript.split())
    hindi_words = sum(len(s.split()) for s, l in segments if l in ["hi", "hi-roman"])
    english_words = sum(len(s.split()) for s, l in segments if l == "en")

    # Normalize if requested
    if normalize_to_native and source_language.startswith("hi"):
        normalized = normalize_code_switched_text(
            transcript,
            target_script="devanagari",
            preserve_english=True,
        )
    else:
        normalized = transcript

    return {
        "original": transcript,
        "normalized": normalized,
        "segments": segments,
        "statistics": {
            "total_words": total_words,
            "hindi_words": hindi_words,
            "english_words": english_words,
            "code_switch_ratio": english_words / total_words if total_words > 0 else 0,
            "num_segments": len(segments),
        },
    }


def merge_code_switched_asr_outputs(
    hindi_transcript: str,
    english_transcript: str,
    timestamps_hindi: Optional[List[Tuple[float, float, str]]] = None,
    timestamps_english: Optional[List[Tuple[float, float, str]]] = None,
) -> str:
    """Merge transcripts from separate Hindi and English ASR.

    Used when running parallel ASR models for different languages.

    Args:
        hindi_transcript: Transcript from Hindi ASR
        english_transcript: Transcript from English ASR
        timestamps_hindi: Optional word timestamps from Hindi ASR
        timestamps_english: Optional word timestamps from English ASR

    Returns:
        Merged transcript
    """
    # If no timestamps, use simple concatenation with normalization
    if timestamps_hindi is None and timestamps_english is None:
        combined = f"{hindi_transcript} {english_transcript}".strip()
        return normalize_code_switched_text(combined, preserve_english=True)

    # Merge based on timestamps
    all_words = []

    if timestamps_hindi:
        for start, end, word in timestamps_hindi:
            all_words.append((start, end, word, "hi"))

    if timestamps_english:
        for start, end, word in timestamps_english:
            all_words.append((start, end, word, "en"))

    # Sort by start time
    all_words.sort(key=lambda x: x[0])

    # Build merged transcript
    result = []
    for _, _, word, lang in all_words:
        if lang == "hi":
            # Keep Hindi as-is (already in native script)
            result.append(word)
        else:
            # Keep English as-is
            result.append(word)

    return " ".join(result)
