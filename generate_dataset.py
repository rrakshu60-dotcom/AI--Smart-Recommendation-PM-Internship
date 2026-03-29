import pandas as pd
import random
import datetime as dt

# -------------------------------------------------------
# Config: titles, companies, locations, categories
# -------------------------------------------------------

titles = [
    "Software Engineering Intern",
    "Backend Developer Intern",
    "Frontend Developer Intern",
    "Full Stack Developer Intern",
    "Data Analyst Intern",
    "Data Science Intern",
    "Machine Learning Intern",
    "AI Research Intern",
    "NLP Intern",
    "Business Analyst Intern",
    "Product Management Intern",
    "Cybersecurity Intern",
    "Cloud Engineering Intern",
    "DevOps Intern",
    "UI/UX Design Intern",
    "Mobile App Developer Intern",
    "QA / Test Automation Intern",
    "Marketing Intern",
    "Growth Intern",
    "Finance Intern"
]

companies = [
    "Google", "Amazon", "Microsoft", "TCS", "Wipro", "Infosys", "Accenture",
    "Flipkart", "Paytm", "IBM", "Byjus", "Swiggy", "Freshworks", "Reliance",
    "Zoho", "Adobe", "Uber", "Salesforce", "Intel", "NVIDIA", "Ola",
    "Zomato", "PhonePe", "Meesho", "CRED"
]

locations = [
    "Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata",
    "Pune", "Noida", "Gurgaon", "Ahmedabad", "Jaipur", "Remote"
]

categories = [
    "Software Development",
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Cybersecurity",
    "Cloud / DevOps",
    "Product",
    "Business / Analytics",
    "Marketing / Growth",
    "Finance",
    "UI/UX Design"
]

# -------------------------------------------------------
# Skills – aligned with your frontend list
# -------------------------------------------------------

core_skills = [
    "Python", "Machine Learning", "Deep Learning", "NLP",
    "Data Analysis", "Statistics", "SQL", "Excel",
    "Pandas", "NumPy", "TensorFlow", "PyTorch",
    "Java", "C++", "C", "HTML/CSS", "JavaScript", "React",
    "REST APIs", "Cloud Computing", "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Linux", "Git/GitHub",
    "Cybersecurity", "Networking",
    "UI/UX Design", "Figma",
    "Power BI", "Tableau",
    "Communication Skills", "Teamwork", "Leadership",
    "Problem Solving", "Time Management", "Creativity",
    "Product Sense", "Marketing Basics", "Finance Basics"
]

# Map categories to anchor skills to make matching sharper
category_to_anchor_skills = {
    "Software Development": [
        "Python", "Java", "C++", "C", "HTML/CSS", "JavaScript",
        "React", "REST APIs", "Git/GitHub", "Problem Solving"
    ],
    "Data Science": [
        "Python", "Data Analysis", "Pandas", "NumPy", "Statistics",
        "SQL", "Power BI", "Tableau", "Machine Learning"
    ],
    "Machine Learning": [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow",
        "PyTorch", "Data Analysis", "Statistics", "NLP"
    ],
    "Artificial Intelligence": [
        "Python", "Machine Learning", "Deep Learning", "NLP",
        "Data Analysis", "TensorFlow", "PyTorch"
    ],
    "Cybersecurity": [
        "Cybersecurity", "Networking", "Linux", "Python",
        "Problem Solving"
    ],
    "Cloud / DevOps": [
        "Cloud Computing", "AWS", "Azure", "GCP",
        "Docker", "Kubernetes", "Linux", "Git/GitHub"
    ],
    "Product": [
        "Product Sense", "Data Analysis", "Communication Skills",
        "Leadership", "Creativity", "Teamwork"
    ],
    "Business / Analytics": [
        "Data Analysis", "Excel", "SQL", "Power BI", "Tableau",
        "Communication Skills"
    ],
    "Marketing / Growth": [
        "Marketing Basics", "Data Analysis", "Creativity",
        "Communication Skills"
    ],
    "Finance": [
        "Finance Basics", "Excel", "Data Analysis",
        "Statistics", "Problem Solving"
    ],
    "UI/UX Design": [
        "UI/UX Design", "Figma", "Creativity",
        "Communication Skills"
    ]
}

# More appealing descriptions
descriptions = [
    "Work on real-time projects used by thousands of users while learning from senior mentors.",
    "Join a fast-paced team, ship features to production and get close guidance from experienced engineers.",
    "Solve practical business problems with data and models, and build a strong portfolio of projects.",
    "Contribute to live ML pipelines, run experiments and present your findings to the product team.",
    "Design and develop user-facing features while collaborating closely with design and product.",
    "Improve your fundamentals with code reviews, pair programming and weekly learning sessions.",
    "Get exposure to industry-standard tools, cloud platforms and modern engineering practices.",
    "Take ownership of a mini project from idea to demo, and present it to leadership at the end.",
    "Work in a student-friendly environment that values learning, feedback and long-term growth.",
    "High-performing interns may be considered for full-time offers or extended internships."
]

# -------------------------------------------------------
# Helper to generate one internship row
# -------------------------------------------------------

def generate_internship_row(today: dt.date):
    title = random.choice(titles)
    company = random.choice(companies)
    location = random.choice(locations)  # "Remote" means WFH, others are on-site
    category = random.choice(categories)

    # Anchor skills from category + some random extras
    anchor = category_to_anchor_skills.get(category, [])
    if anchor:
        anchor_sample = random.sample(anchor, k=min(len(anchor), random.randint(2, 5)))
    else:
        anchor_sample = []
    extra_pool = [s for s in core_skills if s not in anchor_sample]
    extra_sample = random.sample(extra_pool, k=random.randint(1, 3))

    all_skills = anchor_sample + extra_sample
    skills_str = ", ".join(sorted(set(all_skills)))

    desc = random.choice(descriptions)

    # stipend: 30% missing, 70% range value
    if random.random() < 0.3:
        stipend = None
    else:
        low = random.randint(5000, 25000)
        high = random.randint(low + 2000, low + 30000)
        stipend = f"{low}-{high}"

    # Deadline: always in the future (within ~8 months)
    future_days = random.randint(7, 240)
    deadline_date = today + dt.timedelta(days=future_days)
    deadline = deadline_date.strftime("%Y-%m-%d")

    is_remote = "True" if location == "Remote" else "False"

    return [
        title,
        company,
        desc,
        skills_str,
        location,
        stipend,
        category,
        deadline,
        is_remote
    ]

# -------------------------------------------------------
# Generate big dataset
# -------------------------------------------------------

def main():
    num_rows = 1000  # ~10x larger dataset
    today = dt.date.today()

    rows = [generate_internship_row(today) for _ in range(num_rows)]

    df = pd.DataFrame(rows, columns=[
        "title",
        "company_name",
        "description",
        "skills_requirements",
        "location",
        "stipend",
        "category",
        "application_deadline",
        "is_remote"
    ])

    df.to_csv("internships.csv", index=False)
    print(f"Generated internships.csv with {num_rows} rows.")
    print(f"Example deadline range: from >= {today} into the next few months.")

if __name__ == "__main__":
    main()
