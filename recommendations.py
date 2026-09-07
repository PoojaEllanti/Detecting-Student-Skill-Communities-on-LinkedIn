"""
Simple, explainable, similarity/rule based recommendations - deliberately
NOT a complex ML recommender for the first working demo.
"""

from collections import Counter
from .skill_dictionary import SKILL_CATEGORY_MAP


def recommend_related_skills(student_skills, community_skill_lists, top_n=8):
    """Skills that frequently co-occur with this student's skills inside
    their own community, that the student doesn't already have."""
    counter = Counter()
    for skills in community_skill_lists:
        for s in skills:
            if s not in student_skills:
                counter[s] += 1
    return [s for s, _ in counter.most_common(top_n)]


def recommend_similar_students(student_id, similarity_matrix, student_ids, top_n=5):
    idx = student_ids.index(student_id)
    scores = list(enumerate(similarity_matrix[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for j, score in scores:
        if j == idx or score <= 0:
            continue
        results.append((student_ids[j], round(float(score) * 100, 1)))
        if len(results) >= top_n:
            break
    return results


def recommend_communities_for_skills(skills, community_info, top_n=3):
    """Given a raw skill list (e.g. typed by the user), rank communities by
    how much overlap they have with those skills."""
    skill_set = set(skills)
    scored = []
    for cid, info in community_info.items():
        comm_skills = {s for s, _ in info["top_skills"]}
        overlap = len(skill_set & comm_skills)
        if overlap > 0:
            scored.append((cid, info["label"], overlap))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


def related_skill_dictionary_lookup(skill, top_n=6):
    """Skills that share at least one category with the given skill, per
    the curated dictionary (independent of any dataset)."""
    cats = SKILL_CATEGORY_MAP.get(skill.lower().strip())
    if not cats:
        return []
    related = []
    for other_skill, other_cats in SKILL_CATEGORY_MAP.items():
        if other_skill == skill.lower().strip():
            continue
        if set(cats) & set(other_cats):
            related.append(other_skill)
    return sorted(related)[:top_n]
