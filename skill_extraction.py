"""
Hybrid skill extraction:
1. If a structured 'skills' column exists, parse it (comma / semicolon / pipe
   separated) and match each token against the skill dictionary.
2. Always also scan free-text fields (profile_summary, projects,
   certifications, posts, current_role, education) with multi-word,
   longest-phrase-first matching against the same dictionary.
3. Union both sources -> final skill set per student.

This is deliberately dictionary-based (fast, deterministic, no heavy model)
so the first working demo never depends on downloading a transformer.
"""

import re
from .skill_dictionary import ALL_SKILLS

# Common column name candidates -> canonical field name
COLUMN_CANDIDATES = {
    "student_id": ["student_id", "id", "student id", "studentid", "name", "user_id"],
    "profile_summary": ["profile_summary", "summary", "bio", "about", "profile", "headline"],
    "skills": ["skills", "skill", "tech_stack", "technical_skills", "skillset"],
    "projects": ["projects", "project", "portfolio"],
    "certifications": ["certifications", "certification", "certificates", "certs"],
    "internship": ["internship", "internships", "experience"],
    "posts": ["posts", "post", "activity"],
    "education": ["education", "degree", "branch", "major"],
    "current_role": ["current_role", "role", "designation", "title", "current role"],
}


def detect_columns(df_columns):
    """Map actual dataframe columns to canonical fields using fuzzy
    substring matching. Returns dict canonical -> actual_column_name (or
    None if not found)."""
    lower_map = {c.lower().strip(): c for c in df_columns}
    mapping = {}
    for canonical, candidates in COLUMN_CANDIDATES.items():
        found = None
        for cand in candidates:
            if cand in lower_map:
                found = lower_map[cand]
                break
        if not found:
            for lower_col, orig_col in lower_map.items():
                if any(cand in lower_col for cand in candidates):
                    found = orig_col
                    break
        mapping[canonical] = found
    return mapping


_SKILL_PATTERNS = [(s, re.compile(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])")) for s in ALL_SKILLS]


def extract_skills_from_text(text: str):
    """Longest-phrase-first regex matching of the skill dictionary against
    raw (lightly lowercased) text."""
    if not isinstance(text, str) or not text.strip():
        return set()
    t = text.lower()
    found = set()
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(t):
            found.add(skill)
    return found


def parse_structured_skills(raw: str):
    """Parse a delimiter-separated skills string and match each item
    against the dictionary (allows partial/fuzzy containment)."""
    if not isinstance(raw, str) or not raw.strip():
        return set()
    parts = re.split(r"[,;|/]", raw)
    found = set()
    for part in parts:
        p = part.strip().lower()
        if not p:
            continue
        matched_any = False
        for skill in ALL_SKILLS:
            if skill == p or skill in p or p in skill:
                found.add(skill)
                matched_any = True
        if not matched_any and len(p) > 1:
            # keep unknown-but-present raw skill too (normalized) so it can
            # still contribute to similarity even if not in our dictionary
            found.add(p)
    return found


def extract_all_skills_for_row(row, colmap):
    """Combine structured skills column + free text columns for one row."""
    skills = set()

    skills_col = colmap.get("skills")
    if skills_col and skills_col in row and pd_notna(row[skills_col]):
        skills |= parse_structured_skills(str(row[skills_col]))

    text_cols = ["profile_summary", "projects", "certifications", "posts", "current_role", "education"]
    for canon in text_cols:
        col = colmap.get(canon)
        if col and col in row and pd_notna(row[col]):
            skills |= extract_skills_from_text(str(row[col]))

    return sorted(skills)


def pd_notna(value):
    try:
        import pandas as pd
        return pd.notna(value)
    except Exception:
        return value is not None
