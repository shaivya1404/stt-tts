"""Text normalization for Indic languages TTS pipeline.

Provides comprehensive text normalization for:
- Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam
- Number-to-word conversion for Indic numerals
- Abbreviation expansion
- Currency/date formatting
- Punctuation normalization

Dependencies:
    pip install indic-nlp-library
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Try to import indic_nlp_library
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    logger.warning("indic-nlp-library not available. Install with: pip install indic-nlp-library")


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
    "or-IN": "or",
    "pa-IN": "pa",
    "as-IN": "as",
    "en-IN": "en",
}

# Hindi number words
HINDI_ONES = [
    "", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्नासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"
]

HINDI_HUNDRED = "सौ"
HINDI_THOUSAND = "हज़ार"
HINDI_LAKH = "लाख"
HINDI_CRORE = "करोड़"

# Tamil number words
TAMIL_ONES = ["", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
TAMIL_TENS = ["", "பத்து", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு"]
TAMIL_HUNDRED = "நூறு"
TAMIL_THOUSAND = "ஆயிரம்"
TAMIL_LAKH = "லட்சம்"
TAMIL_CRORE = "கோடி"

# Telugu number words
TELUGU_ONES = ["", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
TELUGU_TENS = ["", "పది", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]
TELUGU_HUNDRED = "వంద"
TELUGU_THOUSAND = "వేయి"
TELUGU_LAKH = "లక్ష"
TELUGU_CRORE = "కోటి"

# Bengali number words
BENGALI_ONES = ["", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]
BENGALI_TENS = ["", "দশ", "বিশ", "ত্রিশ", "চল্লিশ", "পঞ্চাশ", "ষাট", "সত্তর", "আশি", "নব্বই"]
BENGALI_HUNDRED = "শত"
BENGALI_THOUSAND = "হাজার"
BENGALI_LAKH = "লক্ষ"
BENGALI_CRORE = "কোটি"

# Common abbreviations for Hindi
HINDI_ABBREVIATIONS: Dict[str, str] = {
    "डॉ.": "डॉक्टर",
    "डॉ": "डॉक्टर",
    "श्री": "श्रीमान",
    "श्रीमती": "श्रीमती",
    "कु.": "कुमारी",
    "प्रो.": "प्रोफेसर",
    "मि.": "मिस्टर",
    "मिसेज": "मिसेज",
    "रु.": "रुपये",
    "₹": "रुपये",
    "कि.मी.": "किलोमीटर",
    "मी.": "मीटर",
    "से.मी.": "सेंटीमीटर",
    "कि.ग्रा.": "किलोग्राम",
    "ग्रा.": "ग्राम",
    "लि.": "लिमिटेड",
    "%": "प्रतिशत",
    "&": "और",
}

# English abbreviations
ENGLISH_ABBREVIATIONS: Dict[str, str] = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Prof.": "Professor",
    "Ltd.": "Limited",
    "Inc.": "Incorporated",
    "Rs.": "Rupees",
    "₹": "Rupees",
    "km": "kilometers",
    "m": "meters",
    "cm": "centimeters",
    "kg": "kilograms",
    "g": "grams",
    "%": "percent",
    "&": "and",
}

# Devanagari to Arabic numeral mapping
DEVANAGARI_NUMERALS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
}

# Tamil numerals
TAMIL_NUMERALS = {
    "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
    "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9"
}


def convert_indic_numerals_to_arabic(text: str) -> str:
    """Convert Indic script numerals to Arabic numerals.

    Args:
        text: Text containing Indic numerals

    Returns:
        Text with Arabic numerals
    """
    result = text
    for indic, arabic in {**DEVANAGARI_NUMERALS, **TAMIL_NUMERALS}.items():
        result = result.replace(indic, arabic)
    return result


def number_to_hindi_words(num: int) -> str:
    """Convert number to Hindi words.

    Args:
        num: Integer number to convert

    Returns:
        Hindi word representation
    """
    if num == 0:
        return "शून्य"

    if num < 0:
        return "ऋण " + number_to_hindi_words(-num)

    if num < 100:
        return HINDI_ONES[num]

    words = []

    # Crores (10 million)
    if num >= 10000000:
        crores = num // 10000000
        words.append(number_to_hindi_words(crores))
        words.append(HINDI_CRORE)
        num %= 10000000

    # Lakhs (100 thousand)
    if num >= 100000:
        lakhs = num // 100000
        words.append(number_to_hindi_words(lakhs))
        words.append(HINDI_LAKH)
        num %= 100000

    # Thousands
    if num >= 1000:
        thousands = num // 1000
        words.append(number_to_hindi_words(thousands))
        words.append(HINDI_THOUSAND)
        num %= 1000

    # Hundreds
    if num >= 100:
        hundreds = num // 100
        words.append(number_to_hindi_words(hundreds))
        words.append(HINDI_HUNDRED)
        num %= 100

    # Remaining (1-99)
    if num > 0:
        words.append(HINDI_ONES[num])

    return " ".join(words)


def number_to_tamil_words(num: int) -> str:
    """Convert number to Tamil words.

    Args:
        num: Integer number to convert

    Returns:
        Tamil word representation
    """
    if num == 0:
        return "சுழியம்"

    if num < 0:
        return "கழித்தல் " + number_to_tamil_words(-num)

    if num < 10:
        return TAMIL_ONES[num]

    if num < 100:
        tens = num // 10
        ones = num % 10
        if ones == 0:
            return TAMIL_TENS[tens]
        return TAMIL_TENS[tens] + " " + TAMIL_ONES[ones]

    words = []

    # Crores
    if num >= 10000000:
        crores = num // 10000000
        words.append(number_to_tamil_words(crores))
        words.append(TAMIL_CRORE)
        num %= 10000000

    # Lakhs
    if num >= 100000:
        lakhs = num // 100000
        words.append(number_to_tamil_words(lakhs))
        words.append(TAMIL_LAKH)
        num %= 100000

    # Thousands
    if num >= 1000:
        thousands = num // 1000
        words.append(number_to_tamil_words(thousands))
        words.append(TAMIL_THOUSAND)
        num %= 1000

    # Hundreds
    if num >= 100:
        hundreds = num // 100
        words.append(number_to_tamil_words(hundreds))
        words.append(TAMIL_HUNDRED)
        num %= 100

    # Remaining
    if num > 0:
        words.append(number_to_tamil_words(num))

    return " ".join(words)


def number_to_telugu_words(num: int) -> str:
    """Convert number to Telugu words.

    Args:
        num: Integer number to convert

    Returns:
        Telugu word representation
    """
    if num == 0:
        return "సున్న"

    if num < 0:
        return "మైనస్ " + number_to_telugu_words(-num)

    if num < 10:
        return TELUGU_ONES[num]

    if num < 100:
        tens = num // 10
        ones = num % 10
        if ones == 0:
            return TELUGU_TENS[tens]
        return TELUGU_TENS[tens] + " " + TELUGU_ONES[ones]

    words = []

    # Crores
    if num >= 10000000:
        crores = num // 10000000
        words.append(number_to_telugu_words(crores))
        words.append(TELUGU_CRORE)
        num %= 10000000

    # Lakhs
    if num >= 100000:
        lakhs = num // 100000
        words.append(number_to_telugu_words(lakhs))
        words.append(TELUGU_LAKH)
        num %= 100000

    # Thousands
    if num >= 1000:
        thousands = num // 1000
        words.append(number_to_telugu_words(thousands))
        words.append(TELUGU_THOUSAND)
        num %= 1000

    # Hundreds
    if num >= 100:
        hundreds = num // 100
        words.append(number_to_telugu_words(hundreds))
        words.append(TELUGU_HUNDRED)
        num %= 100

    # Remaining
    if num > 0:
        words.append(number_to_telugu_words(num))

    return " ".join(words)


def number_to_bengali_words(num: int) -> str:
    """Convert number to Bengali words.

    Args:
        num: Integer number to convert

    Returns:
        Bengali word representation
    """
    if num == 0:
        return "শূন্য"

    if num < 0:
        return "ঋণাত্মক " + number_to_bengali_words(-num)

    if num < 10:
        return BENGALI_ONES[num]

    if num < 100:
        tens = num // 10
        ones = num % 10
        if ones == 0:
            return BENGALI_TENS[tens]
        return BENGALI_TENS[tens] + " " + BENGALI_ONES[ones]

    words = []

    # Crores
    if num >= 10000000:
        crores = num // 10000000
        words.append(number_to_bengali_words(crores))
        words.append(BENGALI_CRORE)
        num %= 10000000

    # Lakhs
    if num >= 100000:
        lakhs = num // 100000
        words.append(number_to_bengali_words(lakhs))
        words.append(BENGALI_LAKH)
        num %= 100000

    # Thousands
    if num >= 1000:
        thousands = num // 1000
        words.append(number_to_bengali_words(thousands))
        words.append(BENGALI_THOUSAND)
        num %= 1000

    # Hundreds
    if num >= 100:
        hundreds = num // 100
        words.append(number_to_bengali_words(hundreds))
        words.append(BENGALI_HUNDRED)
        num %= 100

    # Remaining
    if num > 0:
        words.append(number_to_bengali_words(num))

    return " ".join(words)


def number_to_words(num: int, language: str) -> str:
    """Convert number to words in the specified language.

    Args:
        num: Integer number to convert
        language: Language code

    Returns:
        Word representation
    """
    lang = LANG_MAP.get(language, language)

    if lang == "hi":
        return number_to_hindi_words(num)
    elif lang == "ta":
        return number_to_tamil_words(num)
    elif lang == "te":
        return number_to_telugu_words(num)
    elif lang == "bn":
        return number_to_bengali_words(num)
    else:
        # Fall back to English-style for unsupported languages
        return str(num)


def expand_abbreviations(text: str, language: str) -> str:
    """Expand common abbreviations.

    Args:
        text: Input text
        language: Language code

    Returns:
        Text with expanded abbreviations
    """
    lang = LANG_MAP.get(language, language)

    if lang == "hi":
        abbreviations = HINDI_ABBREVIATIONS
    elif lang == "en":
        abbreviations = ENGLISH_ABBREVIATIONS
    else:
        # Use Hindi abbreviations as default for other Indic languages
        abbreviations = HINDI_ABBREVIATIONS

    result = text
    for abbrev, expansion in abbreviations.items():
        # Use word boundaries for better matching
        result = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, result)
        # Also try without word boundaries for symbols
        if abbrev in "₹%&":
            result = result.replace(abbrev, f" {expansion} ")

    return result


def normalize_currency(text: str, language: str) -> str:
    """Normalize currency expressions.

    Args:
        text: Input text
        language: Language code

    Returns:
        Text with normalized currency
    """
    lang = LANG_MAP.get(language, language)

    # Currency word based on language
    if lang == "hi":
        currency_word = "रुपये"
        paise_word = "पैसे"
    elif lang == "ta":
        currency_word = "ரூபாய்"
        paise_word = "பைசா"
    elif lang == "te":
        currency_word = "రూపాయలు"
        paise_word = "పైసలు"
    elif lang == "bn":
        currency_word = "টাকা"
        paise_word = "পয়সা"
    else:
        currency_word = "rupees"
        paise_word = "paise"

    # Pattern: ₹1,234.56 or Rs. 1234
    def replace_currency(match: re.Match) -> str:
        amount = match.group(1).replace(",", "")
        if "." in amount:
            rupees, paise = amount.split(".")
            rupees = int(rupees) if rupees else 0
            paise = int(paise) if paise else 0
            rupee_words = number_to_words(rupees, language)
            if paise > 0:
                paise_words = number_to_words(paise, language)
                return f"{rupee_words} {currency_word} {paise_words} {paise_word}"
            return f"{rupee_words} {currency_word}"
        else:
            rupees = int(amount)
            rupee_words = number_to_words(rupees, language)
            return f"{rupee_words} {currency_word}"

    # Match currency patterns
    result = re.sub(r'[₹Rs\.]+\s*([\d,]+\.?\d*)', replace_currency, text)
    return result


def normalize_dates(text: str, language: str) -> str:
    """Normalize date expressions.

    Args:
        text: Input text
        language: Language code

    Returns:
        Text with normalized dates
    """
    lang = LANG_MAP.get(language, language)

    # Hindi month names
    hindi_months = [
        "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
        "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"
    ]

    # Pattern: DD/MM/YYYY or DD-MM-YYYY
    def replace_date(match: re.Match) -> str:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        if lang == "hi":
            month_name = hindi_months[month - 1] if 1 <= month <= 12 else str(month)
            day_word = number_to_hindi_words(day)
            year_word = number_to_hindi_words(year)
            return f"{day_word} {month_name} {year_word}"
        else:
            # Default to numeric with words
            day_word = number_to_words(day, language)
            month_word = number_to_words(month, language)
            year_word = number_to_words(year, language)
            return f"{day_word} {month_word} {year_word}"

    result = re.sub(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', replace_date, text)
    return result


def normalize_numbers_in_text(text: str, language: str) -> str:
    """Convert all numbers in text to words.

    Args:
        text: Input text
        language: Language code

    Returns:
        Text with numbers converted to words
    """
    # First convert Indic numerals to Arabic
    text = convert_indic_numerals_to_arabic(text)

    def replace_number(match: re.Match) -> str:
        num_str = match.group(0).replace(",", "")
        try:
            # Handle decimal numbers
            if "." in num_str:
                integer_part, decimal_part = num_str.split(".")
                integer_words = number_to_words(int(integer_part), language)
                # Read decimal digits individually
                decimal_words = " ".join(
                    number_to_words(int(d), language) for d in decimal_part
                )
                lang = LANG_MAP.get(language, language)
                point_word = {
                    "hi": "दशमलव",
                    "ta": "புள்ளி",
                    "te": "పాయింట్",
                    "bn": "দশমিক",
                }.get(lang, "point")
                return f"{integer_words} {point_word} {decimal_words}"
            else:
                return number_to_words(int(num_str), language)
        except (ValueError, IndexError):
            return num_str

    # Match numbers (including with commas and decimals)
    result = re.sub(r'\d[\d,]*\.?\d*', replace_number, text)
    return result


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation marks.

    Args:
        text: Input text

    Returns:
        Text with normalized punctuation
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Normalize different quote styles
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')

    # Remove excessive punctuation
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    return text.strip()


def apply_indic_normalization(text: str, language: str) -> str:
    """Apply indic-nlp-library normalization if available.

    Args:
        text: Input text
        language: Language code

    Returns:
        Normalized text
    """
    if not INDIC_NLP_AVAILABLE:
        return text

    lang = LANG_MAP.get(language, language)

    # Only apply for supported Indic languages
    if lang not in ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "or", "pa", "as"]:
        return text

    try:
        normalizer_factory = IndicNormalizerFactory()
        normalizer = normalizer_factory.get_normalizer(lang)
        return normalizer.normalize(text)
    except Exception as e:
        logger.warning(f"Indic normalization failed: {e}")
        return text


def normalize_text(text: str, language: str) -> str:
    """Normalize text for TTS synthesis.

    Applies comprehensive normalization:
    1. Indic script normalization (if available)
    2. Abbreviation expansion
    3. Currency normalization
    4. Date normalization
    5. Number-to-word conversion
    6. Punctuation normalization

    Args:
        text: Input text to normalize
        language: Target language code (e.g., 'hi-IN', 'ta-IN')

    Returns:
        Normalized text suitable for TTS
    """
    logger.debug("Normalizing text for language {}", language)

    if not text or not text.strip():
        return ""

    # Apply indic normalization first
    result = apply_indic_normalization(text, language)

    # Expand abbreviations
    result = expand_abbreviations(result, language)

    # Normalize currency
    result = normalize_currency(result, language)

    # Normalize dates
    result = normalize_dates(result, language)

    # Convert numbers to words
    result = normalize_numbers_in_text(result, language)

    # Final punctuation cleanup
    result = normalize_punctuation(result)

    logger.debug("Normalized text: {}", result[:100] + "..." if len(result) > 100 else result)

    return result
