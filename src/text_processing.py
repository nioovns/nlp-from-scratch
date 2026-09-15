import re
import os
from nltk.stem import PorterStemmer
from src.file_handler import (
    read_file, write_file
)
STOP_WORDS = set([
    "a","an","the","and","or","but","if","while","with","without",
    "of","at","by","for","from","to","in","on","up","down","over","under",
    "is","am","are","was","were","be","been","being",
    "have","has","had","do","does","did",
    "this","that","these","those",
    "i","you","he","she","it","we","they","me","him","her","them","my","your",
    "his","its","our","their",
    "what","which","who","whom","whose","when","where","why","how"
    ])
OUTPUT_DIR = "./processed_output"


stemmer = PorterStemmer()


def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text) # links and URLs 
    text = re.sub(r'<.*?>', '', text) # HTML tags
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               u"\U0001F1E0-\U0001F1FF"  # Flags
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

def tokenize(text):
    # remaining only english words 
    return re.findall(r'\b[a-zA-Z]+\b', text) 

def lowercase(tokens):
    return [token.lower() for token in tokens]

def word_frequency(tokens):
    frequency = {}
    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1
    return frequency

def stemming(tokens):
    return [stemmer.stem(token) for token in tokens if token.isalpha()]

# removes stopwords to reduce noise
def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOP_WORDS]


def process_text(FILE_PATH):
    text = read_file(FILE_PATH)
    text_clean = clean_text(text)
    
    tokens = tokenize(text_clean)
    write_file("\n".join(tokens), base_name="tokens")
    
    tokens_lower = lowercase(tokens)
    write_file("\n".join(tokens_lower), base_name="lowercase")
    
    no_stopwords = remove_stopwords(tokens_lower)
    write_file("\n".join(no_stopwords), base_name="no_stopwords")
    
    stemmed = stemming(no_stopwords)
    write_file("\n".join(stemmed), base_name="stemmed")
    
    freq = word_frequency(stemmed)
    freq_text = "\n".join([f"{word}: {count}" for word, count in sorted(freq.items(), key=lambda x: x[1], reverse=True)])
    write_file(freq_text, base_name="word_frequency")
    

    return OUTPUT_DIR
