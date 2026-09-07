"""
Lightweight, dependency-safe NLP preprocessing pipeline.

Tries to use NLTK (stopwords + WordNet lemmatizer). If NLTK data isn't
available (no internet at run time), it silently falls back to a built-in
stopword list and a simple suffix-stripping "lemmatizer" so the app NEVER
breaks because of a missing download.
"""

import re

_FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "also", "using", "used", "use", "worked", "work",
    "including", "various", "strong", "good", "experience", "skilled",
}

_NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        STOPWORDS = set(nltk_stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("test")  # forces wordnet lookup
        _NLTK_READY = True
    except LookupError:
        for pkg in ("stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            STOPWORDS = set(nltk_stopwords.words("english"))
            _lemmatizer = WordNetLemmatizer()
            _lemmatizer.lemmatize("test")
            _NLTK_READY = True
        except Exception:
            STOPWORDS = _FALLBACK_STOPWORDS
except Exception:
    STOPWORDS = _FALLBACK_STOPWORDS

STOPWORDS = (STOPWORDS if _NLTK_READY else _FALLBACK_STOPWORDS) | {
    "using", "used", "use", "worked", "work", "including", "various",
}

_SUFFIXES = ("ing", "edly", "ed", "es", "s")


def _simple_lemmatize(token: str) -> str:
    """Very small fallback lemmatizer (suffix stripping) used only when
    NLTK/WordNet data is unavailable."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def lemmatize_token(token: str) -> str:
    if _NLTK_READY:
        try:
            return _lemmatizer.lemmatize(token)
        except Exception:
            return _simple_lemmatize(token)
    return _simple_lemmatize(token)


def clean_text(text: str) -> str:
    """Lowercase -> tokenize -> strip punctuation (preserving tokens like
    c++, c#, node.js) -> remove stopwords -> lemmatize -> rejoin."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    # Preserve a few technical tokens before we strip punctuation
    protected = {
        "c++": "cpluspluslang",
        "c#": "csharplang",
        "node.js": "nodejslang",
        ".net": "dotnetlang",
    }
    for original, placeholder in protected.items():
        text = text.replace(original, placeholder)

    # Keep letters, digits, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()

    restore = {v: k for k, v in protected.items()}
    cleaned_tokens = []
    for tok in tokens:
        if tok in restore:
            cleaned_tokens.append(restore[tok])
            continue
        if tok in STOPWORDS or len(tok) <= 1:
            continue
        cleaned_tokens.append(lemmatize_token(tok))

    return " ".join(cleaned_tokens)


def preprocessing_demo_example():
    """Returns a fixed (before, after) pair for the UI demo panel."""
    before = "I developed a machine learning project using Python and TensorFlow."
    after = clean_text(before)
    return before, after
