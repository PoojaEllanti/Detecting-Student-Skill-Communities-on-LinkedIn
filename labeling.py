"""
AFTER Louvain has discovered communities purely from graph structure, this
module inspects the dominant skills inside each community and assigns a
human-readable label. Louvain never sees these labels or categories - they
are computed strictly as a post-hoc summarization step.
"""

from collections import Counter
from .skill_dictionary import SKILL_CATEGORY_MAP, CATEGORIES


def dominant_skills(skill_lists, top_n=10):
    counter = Counter()
    for skills in skill_lists:
        counter.update(skills)
    return counter.most_common(top_n)


def label_community(skill_lists, top_n_for_scoring=15):
    """Score each known category by (weighted) skill frequency and return
    the best-matching label + the score breakdown."""
    counter = Counter()
    for skills in skill_lists:
        counter.update(skills)

    if not counter:
        return "Mixed / Other", {}

    category_scores = {c: 0.0 for c in CATEGORIES}
    for skill, freq in counter.items():
        cats = SKILL_CATEGORY_MAP.get(skill)
        if not cats:
            continue
        weight = 1.0 / len(cats)
        for c in cats:
            category_scores[c] += freq * weight

    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] <= 0:
        return "Mixed / Other", category_scores

    # If the top two categories are extremely close, call it Mixed
    sorted_scores = sorted(category_scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > 0:
        ratio = sorted_scores[1] / sorted_scores[0] if sorted_scores[0] else 0
        if ratio > 0.9:
            return f"Mixed ({best_category} leaning)", category_scores

    return best_category, category_scores


def label_all_communities(partition, student_skills_map):
    """partition: node -> community_id
    student_skills_map: node -> list of skills
    Returns: dict community_id -> {label, top_skills, size, students}
    """
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    result = {}
    for cid, members in communities.items():
        skill_lists = [student_skills_map.get(m, []) for m in members]
        label, scores = label_community(skill_lists)
        top_skills = dominant_skills(skill_lists, top_n=10)
        result[cid] = {
            "label": label,
            "top_skills": top_skills,
            "size": len(members),
            "students": members,
            "category_scores": scores,
        }
    return result
