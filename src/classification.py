import os
import math
from collections import Counter
from src.text_processing import (
    tokenize, lowercase,
    remove_stopwords, clean_text, word_frequency
)
from src.file_handler import (
    read_file
)

def preprocess(text):
    text = clean_text(text)
    tokens = lowercase(tokenize(text))
    tokens = remove_stopwords(tokens)
    return tokens

def load_dataset(path):
    classes = [] # name of classes
    vocab = set() # vocabulary - set of all unique words
    docs_cache = {}  # cache for storing tokenized docs
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        # skip test folders and non-directories
        if not os.path.isdir(class_path) or class_name.endswith("-test"):
            continue
        classes.append(class_name) # add class name 
        #loop over documents of each class
        for file_name in os.listdir(class_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(class_path, file_name)
                tokens = preprocess(read_file(file_path)) # read and preprocess
                vocab.update(tokens) # update vocabulary
                docs_cache[file_path] = tokens # cache tokens
    return classes, vocab, docs_cache

def train_naive_bayes(path, classes, vocab, docs_cache):
    word_counts_per_class = {} # word frequency per class
    total_words_per_class = {} # total number of words per class 
    doc_count_per_class = {} # number of docs per class
    total_docs = 0 

    log_P_w_c = {}
    for c in classes:
        word_counts = Counter()
        total_words = 0
        doc_count = 0
        class_path = os.path.join(path, c)
        for file_name in os.listdir(class_path):
            if file_name.endswith(".txt"):
                doc_count += 1
                total_docs += 1
                file_path = os.path.join(class_path, file_name)
                tokens = docs_cache[file_path] 
                counts = word_frequency(tokens)
                word_counts.update(counts)
                total_words += sum(counts.values())

        word_counts_per_class[c] = word_counts
        total_words_per_class[c] = total_words
        doc_count_per_class[c] = doc_count

    # P(c) - prior probability
    P_c = {c: doc_count_per_class[c] / total_docs for c in classes}
    
    # vocabulary size
    V = len(vocab)

    # log(P(w|c)) - likelihood with Laplace smoothing
    for c in classes:
        log_P_w_c[c] = {w: math.log((word_counts_per_class[c][w]+1)/(total_words_per_class[c]+V)) for w in vocab}

    return {"classes": classes, "vocab": vocab, "P_c": P_c, "log_P_w_c": log_P_w_c}

# goal: maximum p(c|d)
# p(c|d) = p(c).Пp(w|c)^freq(w,d) 
# to prevent underflow:
# p(c|d) = log p(c) + ∑ freq(w,d) . log p(w|c)
def predict(model, text):
    tokens = preprocess(text)
    counts = word_frequency(tokens)
    scores = {}
    for c in model["classes"]:
        score = math.log(model["P_c"][c])
        for w, freq in counts.items():
            if w in model["vocab"]:
                score += freq * model["log_P_w_c"][c][w] 
        scores[c] = score
    best_class = max(scores, key=scores.get)
    return best_class

# accuracy = correct / total 
# precision = Tp / TP + FP
# recall = TP / TP + FN
# F1-score = 2 * precision * recall / (precision + recall)
def evaluate(model, classes, data_path, docs_cache):
    class_index = {c:i for i,c in enumerate(classes)}
    n = len(classes)
    confusion = [[0]*n for _ in range(n)]
    total, correct = 0, 0

    for true_class in classes:
        test_path = os.path.join(data_path, true_class, "test")
        if not os.path.isdir(test_path):
            continue
        for file_name in os.listdir(test_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(test_path, file_name)
                if file_path in docs_cache:
                    tokens = docs_cache[file_path]
                else:
                    tokens = preprocess(read_file(file_path))
                pred_class = predict(model, " ".join(tokens))
                i, j = class_index[true_class], class_index[pred_class]
                confusion[i][j] += 1
                total += 1
                if true_class == pred_class:
                    correct += 1

    accuracy = correct / total if total > 0 else 0
    full_text=f"Accuracy: {accuracy:.2%}\n"
    col_width = max([len(c) for c in classes] + [7])+ 2
    
    for idx, cls in enumerate(classes):
        TP = confusion[idx][idx]
        FP = sum(confusion[r][idx] for r in range(n))-TP
        FN = sum(confusion[idx])-TP
        precision = TP/(TP+FP) if (TP+FP)>0 else 0
        recall = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0
        full_text += f"{cls:<{col_width}} Precision: {precision:.2%}  Recall: {recall:.2%}  F1: {f1:.2%}\n"
    return full_text

# class for training and prediction
class ClassificationModel:
    def __init__(self, DATASET_PATH='./datasets/Classification-Train And Test'):
        #load- train - evaluate
        print("Loading classification model...")
        classes, vocab, docs_cache = load_dataset(DATASET_PATH)
        self.model = train_naive_bayes(DATASET_PATH, classes, vocab, docs_cache)
        self.evaluation_result = evaluate(self.model, classes, DATASET_PATH, docs_cache)
            
    def predict(self, test_file):
        return predict(self.model, read_file(test_file)), self.evaluation_result
