"""
Student representation + similarity.

Each student's skill set is turned into a "document" (skills joined by
spaces, with underscores inside multi-word skills so they act as single
tokens). TF-IDF is then fit across all students, and cosine similarity is
computed between every pair.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def skills_to_document(skills):
    return " ".join(s.replace(" ", "_") for s in skills)


def build_tfidf_matrix(list_of_skill_lists):
    docs = [skills_to_document(s) if s else "no_skills_listed" for s in list_of_skill_lists]
    vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
    matrix = vectorizer.fit_transform(docs)
    return matrix, vectorizer


def compute_similarity_matrix(list_of_skill_lists):
    matrix, vectorizer = build_tfidf_matrix(list_of_skill_lists)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)  # no self-loops
    return sim, vectorizer


def similarity_demo_example(skills_a, skills_b, skills_c=None):
    """Small illustrative example for the UI: similarity between two/three
    example skill sets."""
    lists = [skills_a, skills_b] + ([skills_c] if skills_c else [])
    sim, _ = compute_similarity_matrix(lists)
    return sim
