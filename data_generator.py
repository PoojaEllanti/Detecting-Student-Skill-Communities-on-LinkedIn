"""
Generates a synthetic-but-realistic student profile dataset with several
skill clusters that OVERLAP (e.g. some "Data Science" students also know
some "AI/ML" skills, some "Web Development" students dabble in "Cloud").

We deliberately do NOT write a ground-truth community column into the CSV
that the app reads for detection - communities are always discovered by
Louvain on the graph. The cluster used for generation is only used to make
skill co-occurrence realistic.
"""

import random

CLUSTER_CORE_SKILLS = {
    "AI/ML": ["python", "machine learning", "deep learning", "tensorflow",
              "pytorch", "nlp", "neural networks", "scikit-learn"],
    "Data Science": ["python", "pandas", "numpy", "sql", "power bi",
                      "data analysis", "statistics", "excel"],
    "Web Development": ["html", "css", "javascript", "react", "node.js",
                          "mongodb", "rest api", "typescript"],
    "Cloud/DevOps": ["aws", "docker", "kubernetes", "linux", "terraform",
                       "ci/cd", "git", "devops"],
    "Cybersecurity": ["cybersecurity", "networking", "ethical hacking",
                        "linux", "cryptography", "network security", "git"],
}

# Skills a cluster occasionally borrows from a neighbouring cluster -
# this is what creates realistic overlap for Louvain to resolve.
CLUSTER_OVERLAP_SKILLS = {
    "AI/ML": ["sql", "docker", "statistics"],
    "Data Science": ["machine learning", "power bi", "sql"],
    "Web Development": ["aws", "docker", "git"],
    "Cloud/DevOps": ["python", "networking", "linux"],
    "Cybersecurity": ["python", "aws", "networking"],
}

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Kabir", "Rohan", "Sai",
               "Ananya", "Diya", "Isha", "Kavya", "Meera", "Priya", "Sara",
               "Zara", "Arjun", "Dev", "Karthik", "Nikhil", "Rahul", "Riya",
               "Sneha", "Tanvi", "Varun", "Yash", "Aisha", "Neha", "Pooja",
               "Rhea", "Simran"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Gupta",
              "Menon", "Rao", "Singh", "Kapoor", "Das", "Bose", "Chatterjee",
              "Pillai"]

PROJECT_TEMPLATES = {
    "AI/ML": [
        "Built a {skill1} model to classify images using {skill2}",
        "Developed a chatbot leveraging {skill1} and {skill2} techniques",
        "Trained a {skill1} pipeline for sentiment analysis with {skill2}",
    ],
    "Data Science": [
        "Analyzed sales data using {skill1} and visualized insights in {skill2}",
        "Built an interactive dashboard with {skill1} on top of a {skill2} pipeline",
        "Cleaned and modeled a large dataset using {skill1} and {skill2}",
    ],
    "Web Development": [
        "Built a full-stack web app using {skill1} and {skill2}",
        "Developed a responsive e-commerce site with {skill1} and {skill2}",
        "Created a REST API backend using {skill1} integrated with {skill2}",
    ],
    "Cloud/DevOps": [
        "Deployed a microservices architecture using {skill1} and {skill2}",
        "Automated CI/CD pipelines with {skill1} on {skill2} infrastructure",
        "Containerized applications using {skill1} orchestrated with {skill2}",
    ],
    "Cybersecurity": [
        "Performed a penetration test using {skill1} and documented {skill2} findings",
        "Set up a secure network lab practicing {skill1} and {skill2}",
        "Built a intrusion detection tool combining {skill1} and {skill2}",
    ],
}

SUMMARY_TEMPLATES = [
    "Final-year engineering student passionate about {c}. Enjoys building projects with {s1} and {s2}.",
    "Aspiring {c} professional with hands-on experience in {s1} and {s2} through college projects.",
    "Enthusiastic learner exploring {c}, currently strengthening skills in {s1} and {s2}.",
    "Motivated student focused on {c}, has worked on internships involving {s1} and {s2}.",
]

CERT_TEMPLATES = {
    "AI/ML": ["Deep Learning Specialization", "TensorFlow Developer Certificate", "Machine Learning by Andrew Ng"],
    "Data Science": ["Google Data Analytics Certificate", "IBM Data Science Professional Certificate", "Power BI Data Analyst Associate"],
    "Web Development": ["Meta Front-End Developer Certificate", "The Complete Web Developer Bootcamp", "React - The Complete Guide"],
    "Cloud/DevOps": ["AWS Certified Cloud Practitioner", "Docker & Kubernetes Certification", "HashiCorp Terraform Associate"],
    "Cybersecurity": ["CompTIA Security+", "Certified Ethical Hacker (CEH)", "Google Cybersecurity Certificate"],
}

EDUCATION_OPTIONS = [
    "B.Tech Computer Science Engineering",
    "B.Tech Information Technology",
    "B.E Computer Science",
    "B.Tech Electronics and Communication",
    "B.Sc Computer Science",
]

ROLE_TEMPLATES = {
    "AI/ML": ["ML Intern", "AI Research Intern", "Data Science Trainee"],
    "Data Science": ["Data Analyst Intern", "BI Intern", "Data Science Intern"],
    "Web Development": ["Frontend Developer Intern", "Full-Stack Intern", "Web Developer Trainee"],
    "Cloud/DevOps": ["Cloud Intern", "DevOps Trainee", "Site Reliability Intern"],
    "Cybersecurity": ["Security Analyst Intern", "SOC Trainee", "Cybersecurity Intern"],
}


def _pick_skills(cluster, rng, n_core=5, n_overlap_chance=0.5):
    core = CLUSTER_CORE_SKILLS[cluster]
    chosen = set(rng.sample(core, k=min(n_core, len(core))))
    for s in CLUSTER_OVERLAP_SKILLS[cluster]:
        if rng.random() < n_overlap_chance:
            chosen.add(s)
    # small chance of borrowing one skill from a totally different cluster
    if rng.random() < 0.15:
        other_cluster = rng.choice([c for c in CLUSTER_CORE_SKILLS if c != cluster])
        chosen.add(rng.choice(CLUSTER_CORE_SKILLS[other_cluster]))
    return sorted(chosen)


def generate_demo_dataset(n_students=90, seed=42):
    rng = random.Random(seed)
    clusters = list(CLUSTER_CORE_SKILLS.keys())
    rows = []

    for i in range(1, n_students + 1):
        cluster = clusters[(i - 1) % len(clusters)]
        # shuffle slightly so cluster order isn't perfectly repeating
        if rng.random() < 0.1:
            cluster = rng.choice(clusters)

        skills = _pick_skills(cluster, rng)
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        student_id = f"S{i:03d}"

        s1, s2 = (skills[0], skills[1]) if len(skills) >= 2 else (skills[0], skills[0])
        summary = rng.choice(SUMMARY_TEMPLATES).format(c=cluster, s1=s1, s2=s2)

        proj_template = rng.choice(PROJECT_TEMPLATES[cluster])
        ps1, ps2 = rng.sample(skills, 2) if len(skills) >= 2 else (skills[0], skills[0])
        project = proj_template.format(skill1=ps1, skill2=ps2)

        cert = rng.choice(CERT_TEMPLATES[cluster]) if rng.random() < 0.7 else ""
        role = rng.choice(ROLE_TEMPLATES[cluster]) if rng.random() < 0.6 else ""
        education = rng.choice(EDUCATION_OPTIONS)
        post = f"Excited to share I completed a project on {cluster.lower()} using {s1}! #{s1.replace(' ', '')}" if rng.random() < 0.5 else ""

        rows.append({
            "student_id": student_id,
            "name": name,
            "profile_summary": summary,
            "skills": ", ".join(skills),
            "projects": project,
            "certifications": cert,
            "internship": role,
            "posts": post,
            "education": education,
            "current_role": role if role else "Student",
        })

    return rows


def save_demo_csv(path, n_students=90, seed=42):
    import csv
    rows = generate_demo_dataset(n_students=n_students, seed=seed)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    save_demo_csv("data/demo_students.csv")
    print("Demo dataset written to data/demo_students.csv")
