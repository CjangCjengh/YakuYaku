import re

def basic_cleaner(text):
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    return text
