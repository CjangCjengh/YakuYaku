import re
import opencc

def basic_cleaner(text):
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    return text

def s2t(text):
    return opencc.OpenCC('s2tw').convert(text)

def t2s(text):
    return opencc.OpenCC('t2s').convert(text)
