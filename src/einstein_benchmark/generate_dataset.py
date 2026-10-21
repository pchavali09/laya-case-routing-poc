"""
Einstein Benchmark Dataset Generator
Generates 1,000 unique cases (800 train, 200 test) across 5 departments:
Billing, Technical, Sales, Operations, Other.

Rules enforced:
- 160 train, 40 test per department (800 train, 200 test total)
- Test difficulty: 16 easy (40%), 12 paraphrase (30%), 8 multi_topic (20%), 4 ambiguous (10%)
- Train difficulty: 80 easy (50%), 40 paraphrase (25%), 24 multi_topic (15%), 16 ambiguous (10%)
- Voices mixed: angry, polite, terse, rambling, non_native, forwarded, typo, agent_note
- Subject: 4 to 15 words, does NOT start with department name or obvious label
- Description: 2 to 8 sentences
- Origin: mix of Email, Web, Phone
- Priority: Low, Medium, High
- Max TF-IDF cosine similarity <= 0.85
- Completely separate train vs test scenarios
"""

import os
import re
import csv
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

def count_words(text):
    return len(text.strip().split())

def count_sentences(text):
    # Strip common abbreviations that cause false sentence breaks
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|etc\.|v\d+\.\d+|no\.|ref\.|dr\.|mr\.|ms\.|corp\.|inc\.|ltd\.)', '', text, flags=re.IGNORECASE)
    sents = [s.strip() for s in re.split(r'[.!?]+(?:\s+|$)', cleaned) if len(s.strip()) > 1]
    return len(sents)

def clean_subject(subj):
    # Ensure subject doesn't start with department names or obvious tags
    words = subj.strip().split()
    first = words[0].rstrip(":,.-").lower()
    if first in ["billing", "technical", "sales", "operations", "other", "tech", "ops", "finance", "support", "ticket"]:
        words = ["Regarding"] + words[1:]
    return " ".join(words)
