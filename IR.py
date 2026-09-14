import os
import math
from collections import Counter
from TextProcessing import (
    read_file, tokenize, lowercase, 
    remove_stopwords, clean_text
)

def preprocess(text):
    text = clean_text(text)
    tokens = lowercase(tokenize(text))
    tokens = remove_stopwords(tokens)
    return tokens

def load_documents(path):
    docs = {} # dictionary to store every document tokens
    for file in os.listdir(path):
        if file.endswith(".txt"):
            docs[file] = preprocess(read_file(os.path.join(path, file))) 
    return docs

def compute_tf_df(docs):
    tf = {}  # term frequency for each document
    df = Counter() # document frequency for each term
    for doc_name, tokens in docs.items():
        counter = Counter(tokens)
        tf[doc_name] = counter
        for term in counter:
            df[term] += 1
    return tf, df

# Sparse dictionary-based representation to reduce memory usage
def compute_tfidf(tf, df, N):
    # compute IDF values: log N/df(term) 
    # tf-idf weighting: (1+log tf(term,d)).log N/df(term) 
    idf = {term: math.log10(N / df[term]) for term in df}  
    tfidf_docs = {}
    norms = {}  
    for doc, counter in tf.items():
        doc_tfidf = {}
        sq_sum = 0
        for term, freq in counter.items():
            #tf-idf weighting
            w = (1 + math.log10(freq)) * idf[term]
            doc_tfidf[term] = w
            sq_sum += w**2
        tfidf_docs[doc] = doc_tfidf
        #vector norm for cosine similarity
        norms[doc] = math.sqrt(sq_sum)
    return tfidf_docs, idf, norms

def query_vector(tokens, idf):
    # build sparse tf-idf vector for query
    counter = Counter(tokens)
    vec = {t: (1 + math.log10(f)) * idf[t] for t, f in counter.items() if t in idf}
    return vec

# cosine = q.d / norm(q) * norm(d) 
def cosine(q_vec, doc_vec, doc_norm, q_norm=None):
    # compute cosine similarity between query and doc
    if not q_vec or not doc_vec:
        return 0.0
    common_terms = q_vec.keys() & doc_vec.keys()
    if not common_terms:
        return 0.0
    # dot product
    num = sum(q_vec[t] * doc_vec[t] for t in common_terms)
    if q_norm is None:
        q_norm = math.sqrt(sum(v**2 for v in q_vec.values()))
    return num / (q_norm * doc_norm) if q_norm*doc_norm != 0 else 0.0

def score_tfidf_sum(q_tokens, doc_vec):
    # Score documents by summing tf-idf weights of query terms
    return sum(doc_vec.get(t, 0) for t in q_tokens)

class IRModel:
    def __init__(self, dataset_path="./datasets/IR/IR Documents to Index"):
        #load - compute tf-idf and norms 
        print("Loading IR model...")
        self.docs_tokens = load_documents(dataset_path)
        self.tf, self.df = compute_tf_df(self.docs_tokens)
        self.N = len(self.tf)
        self.tfidf_docs, self.idf, self.norms = compute_tfidf(self.tf, self.df, self.N)
        
    def query(self, q, method="vsm", top_k=10):
        q_tokens = preprocess(q)
        if method == "tfidf_sum":
            # rank docs using tf-idf sum method
            ranked = [(doc, score_tfidf_sum(q_tokens, doc_vec))
                      for doc, doc_vec in self.tfidf_docs.items()]
            ranked.sort(key=lambda x: x[1], reverse=True)

        elif method == "vsm":
            # Vector Space Model (VSM) with cosine similarity
            q_vec = query_vector(q_tokens, self.idf)
            q_norm = math.sqrt(sum(v**2 for v in q_vec.values()))
            ranked = [(doc, cosine(q_vec, doc_vec, self.norms[doc], q_norm))
                      for doc, doc_vec in self.tfidf_docs.items()]
            ranked.sort(key=lambda x: x[1], reverse=True)

        else:
            raise ValueError("method must be 'vsm' or 'tfidf_sum'")

        return ranked[:top_k]
