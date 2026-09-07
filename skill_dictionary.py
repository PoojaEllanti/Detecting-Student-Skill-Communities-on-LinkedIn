"""
Curated technical skill dictionary + category mapping.

Each skill maps to one or more categories (a skill can be shared across
categories, e.g. 'python' or 'sql', which is realistic and is exactly what
gives Louvain overlapping structure to actually resolve).
"""

SKILL_CATEGORY_MAP = {
    # ---------------- AI / ML ----------------
    "machine learning": ["AI/ML"],
    "deep learning": ["AI/ML"],
    "artificial intelligence": ["AI/ML"],
    "neural networks": ["AI/ML"],
    "natural language processing": ["AI/ML"],
    "nlp": ["AI/ML"],
    "computer vision": ["AI/ML"],
    "tensorflow": ["AI/ML"],
    "pytorch": ["AI/ML"],
    "keras": ["AI/ML"],
    "scikit-learn": ["AI/ML", "Data Science"],
    "opencv": ["AI/ML"],
    "reinforcement learning": ["AI/ML"],
    "generative ai": ["AI/ML"],
    "llm": ["AI/ML"],
    "transformers": ["AI/ML"],

    # ---------------- Data Science ----------------
    "data science": ["Data Science"],
    "data analysis": ["Data Science"],
    "data visualization": ["Data Science"],
    "pandas": ["Data Science", "AI/ML"],
    "numpy": ["Data Science", "AI/ML"],
    "power bi": ["Data Science"],
    "tableau": ["Data Science"],
    "excel": ["Data Science"],
    "statistics": ["Data Science"],
    "r programming": ["Data Science"],
    "big data": ["Data Science"],
    "spark": ["Data Science"],
    "hadoop": ["Data Science"],
    "etl": ["Data Science"],

    # ---------------- Web Development ----------------
    "html": ["Web Development"],
    "css": ["Web Development"],
    "javascript": ["Web Development"],
    "typescript": ["Web Development"],
    "react": ["Web Development"],
    "node.js": ["Web Development"],
    "angular": ["Web Development"],
    "vue": ["Web Development"],
    "django": ["Web Development"],
    "flask": ["Web Development"],
    "php": ["Web Development"],
    "bootstrap": ["Web Development"],
    "rest api": ["Web Development"],
    "next.js": ["Web Development"],
    "express.js": ["Web Development"],
    "mongodb": ["Web Development", "Cloud/DevOps"],
    "mysql": ["Web Development", "Data Science"],
    "spring boot": ["Web Development"],

    # ---------------- Cloud / DevOps ----------------
    "aws": ["Cloud/DevOps"],
    "azure": ["Cloud/DevOps"],
    "gcp": ["Cloud/DevOps"],
    "docker": ["Cloud/DevOps"],
    "kubernetes": ["Cloud/DevOps"],
    "terraform": ["Cloud/DevOps"],
    "jenkins": ["Cloud/DevOps"],
    "ci/cd": ["Cloud/DevOps"],
    "linux": ["Cloud/DevOps", "Cybersecurity"],
    "devops": ["Cloud/DevOps"],
    "cloud computing": ["Cloud/DevOps"],
    "ansible": ["Cloud/DevOps"],
    "microservices": ["Cloud/DevOps", "Web Development"],
    "git": ["Cloud/DevOps", "Web Development"],

    # ---------------- Cybersecurity ----------------
    "cybersecurity": ["Cybersecurity"],
    "networking": ["Cybersecurity", "Cloud/DevOps"],
    "ethical hacking": ["Cybersecurity"],
    "penetration testing": ["Cybersecurity"],
    "cryptography": ["Cybersecurity"],
    "network security": ["Cybersecurity"],
    "firewall": ["Cybersecurity"],
    "malware analysis": ["Cybersecurity"],
    "siem": ["Cybersecurity"],
    "vulnerability assessment": ["Cybersecurity"],

    # ---------------- General / cross-cutting languages ----------------
    "python": ["AI/ML", "Data Science", "Web Development", "Cloud/DevOps"],
    "java": ["Web Development", "Cloud/DevOps"],
    "c++": ["AI/ML", "Cybersecurity"],
    "c#": ["Web Development"],
    "sql": ["Data Science", "Web Development"],
    "matlab": ["AI/ML", "Data Science"],
}

# Flat list used for phrase matching, longest-phrase-first so multi-word
# skills are matched before their sub-tokens.
ALL_SKILLS = sorted(SKILL_CATEGORY_MAP.keys(), key=len, reverse=True)

CATEGORIES = ["AI/ML", "Data Science", "Web Development", "Cloud/DevOps", "Cybersecurity"]

CATEGORY_COLORS = {
    "AI/ML": "#6C5CE7",
    "Data Science": "#00B894",
    "Web Development": "#0984E3",
    "Cloud/DevOps": "#E17055",
    "Cybersecurity": "#D63031",
    "Mixed / Other": "#636E72",
}
