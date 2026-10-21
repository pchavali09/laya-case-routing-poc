"""
Full Benchmark Builder for Einstein Case Classification
Generates 1,000 unique cases (800 train, 200 test) across 5 departments:
Billing, Technical, Sales, Operations, Other.
"""
import os
import re
import csv
import json
import random
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

def count_words(text):
    return len(text.strip().split())

def count_sentences(text):
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|etc\.|v\d+\.\d+|no\.|ref\.|dr\.|mr\.|ms\.|corp\.|inc\.|ltd\.)', '', text, flags=re.IGNORECASE)
    sents = [s.strip() for s in re.split(r'[.!?]+(?:\s+|$)', cleaned) if len(s.strip()) > 1]
    return len(sents)

def check_first_word(subj):
    first_w = subj.strip().split()[0].rstrip(":,.-").lower()
    prohibited = ["billing", "technical", "sales", "operations", "other", "tech", "ops", "finance"]
    if first_w in prohibited:
        return False, first_w
    return True, first_w

print("Helper utilities loaded.")
