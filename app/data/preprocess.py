import re
import html

def clean_text(text: str) -> str:
    """Clean and normalize text for embedding"""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)

    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()