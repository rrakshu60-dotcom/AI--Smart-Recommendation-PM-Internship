from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow index.html on file:// to call backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# MODELS
# ======================================================
class User(BaseModel):
    name: str
    age: int
    email: str
    phone: str
    employment_status: str
    is_full_time_student: bool
    family_annual_income: float
    family_has_govt_job: bool
    education_level: str
    skills: List[str]
    location: str
    preferred_locations: List[str]
    interests: List[str]
    previous_experience: Optional[str] = None


# ======================================================
# RECOMMENDATION ENGINE
# ======================================================
class RecommendationEngine:
    def __init__(self, dataset_path: str = "internships.csv"):
        self.users = []

        try:
            self.df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

        # Fill missing values
        self.df = self.df.fillna("")

        # Normalise is_remote to bool
        def to_bool(x):
            if isinstance(x, str):
                return x.strip().lower() in ["true", "1", "yes"]
            return bool(x)

        if "is_remote" in self.df.columns:
            self.df["is_remote_bool"] = self.df["is_remote"].apply(to_bool)
        else:
            self.df["is_remote_bool"] = False

        # Combine text features for TF-IDF
        text_cols = ["title", "description", "skills_requirements", "category", "location"]
        for col in text_cols:
            if col not in self.df.columns:
                self.df[col] = ""

        self.df["text_features"] = (
            self.df["title"].astype(str) + " " +
            self.df["description"].astype(str) + " " +
            self.df["skills_requirements"].astype(str) + " " +
            self.df["category"].astype(str) + " " +
            self.df["location"].astype(str)
        )

        # TF-IDF on text features
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text_features"])

        # Process stipend
        def clean_stipend_value(raw):
            # your CSV has stipend either None or "low-high"
            if raw is None or raw == "":
                return 0.0
            if isinstance(raw, (int, float)):
                return float(raw)
            try:
                s = str(raw)
                if "-" in s:
                    low = s.split("-")[0].strip()
                    return float(low)
                return float(s)
            except Exception:
                return 0.0

        self.df["stipend_numeric"] = self.df["stipend"].apply(clean_stipend_value)

        # Scale stipend (0–1)
        self.scaler = MinMaxScaler()
        self.df["stipend_scaled"] = self.scaler.fit_transform(
            self.df[["stipend_numeric"]]
        )

        # Pre-compute skills as a list (lowercased)
        def split_skills(s):
            return [
                part.strip().lower()
                for part in str(s).split(",")
                if part.strip()
            ]

        self.df["skills_list"] = self.df["skills_requirements"].apply(split_skills)

        # Build numeric feature matrix: TF-IDF + stipend + remote flag
        extra_features = np.vstack([
            self.df["stipend_scaled"].values,
            self.df["is_remote_bool"].astype(int).values
        ]).T  # shape (n_samples, 2)

        self.internship_features = np.hstack([
            self.tfidf_matrix.toarray(),
            extra_features
        ])

    def register_user(self, user_dict: dict) -> int:
        """Store user in memory and return user_id (1-based index)."""
        self.users.append(user_dict)
        return len(self.users)

    def build_user_vector(self, user: dict) -> np.ndarray:
        """Vectorise user profile text to same space as internships."""
        skills = [s.lower() for s in user.get("skills", [])]
        interests = [i.lower() for i in user.get("interests", [])]

        # Repeat skills to give them extra weight in TF-IDF
        skill_text = " ".join(skills)
        weighted_skill_text = (" " + skill_text) * 3 if skill_text else ""

        profile_text = (
            weighted_skill_text + " " +
            " ".join(interests) + " " +
            str(user.get("location", "")) + " " +
            str(user.get("education_level", ""))
        )

        user_vec = self.vectorizer.transform([profile_text]).toarray()

        # Neutral extra features
        extra = np.array([[0.5, 0.5]])  # stipend preference, remote preference
        return np.hstack([user_vec, extra])

    def _compute_scores(self, user: dict, user_vector: np.ndarray):
        """Compute similarity + skill/location bonuses for all internships."""
        # Base cosine similarity
        sims = cosine_similarity(user_vector, self.internship_features)[0]

        user_skills_lower = [s.lower() for s in user.get("skills", []) if s.strip()]
        user_prefs_lower = [p.lower() for p in user.get("preferred_locations", []) if p.strip()]
        user_city_lower = str(user.get("location", "")).lower()

        scores = []

        for idx, row in self.df.iterrows():
            rec_skills = row["skills_list"]
            matched_skills = sorted(set(user_skills_lower) & set(rec_skills))

            # Skill overlap: fraction of user's skills appearing in this internship
            if user_skills_lower:
                skill_score = len(matched_skills) / len(user_skills_lower)
            else:
                skill_score = 0.0

            # Location / remote preference score
            loc_score = 0.0
            rec_city = str(row["location"]).lower()
            is_remote = bool(row["is_remote_bool"])

            if rec_city == user_city_lower and rec_city != "remote":
                loc_score = 1.0
            elif rec_city in user_prefs_lower and rec_city != "remote":
                loc_score = 0.8
            elif is_remote and ("remote" in user_prefs_lower or not user_prefs_lower):
                loc_score = 0.7

            base_sim = sims[idx]

            # Final score: base similarity + 0.25 * skill match + 0.15 * location match
            final_score = base_sim + 0.25 * skill_score + 0.15 * loc_score

            scores.append({
                "index": idx,
                "final_score": final_score,
                "base_sim": base_sim,
                "skill_score": skill_score,
                "loc_score": loc_score,
                "matched_skills": matched_skills
            })

        return scores

    def get_recommendations(self, user_id: int, limit: int = 5):
        user_idx = user_id - 1
        if user_idx < 0 or user_idx >= len(self.users):
            return {"eligible": True, "recommendations": [], "message": "No profile found."}

        user = self.users[user_idx]
        user_vector = self.build_user_vector(user)

        score_rows = self._compute_scores(user, user_vector)

        # Sort by our final_score descending
        score_rows.sort(key=lambda r: r["final_score"], reverse=True)

        # Build response list
        recommendations = []
        for rank, row_info in enumerate(score_rows[:limit], start=1):
            idx = row_info["index"]
            row = self.df.iloc[idx]

            stipend_numeric = float(row["stipend_numeric"])
            raw_stipend = row.get("stipend", "")
            is_remote = bool(row["is_remote_bool"])
            matched_skills = row_info["matched_skills"]

            reasons = []
            if matched_skills:
                # Make them pretty (capitalise first letter)
                pretty = [s.capitalize() for s in matched_skills[:3]]
                reasons.append("Matches your skills: " + ", ".join(pretty))
            else:
                reasons.append("Relevant to your skills and profile")

            if row_info["loc_score"] > 0:
                reasons.append("Location or remote preference aligned")

            if stipend_numeric > 0:
                reasons.append("Stipend available for this role")

            recommendations.append({
                "rank": rank,
                "title": row["title"],
                "company_name": row["company_name"],
                "description": row["description"],
                "skills_requirements": row["skills_requirements"],
                "location": row["location"],
                "raw_location": row["location"],
                "stipend": stipend_numeric,
                "raw_stipend": raw_stipend,
                "category": row["category"],
                "application_deadline": row["application_deadline"],
                "is_remote": is_remote,
                "reasons": reasons,
            })

        return {
            "eligible": True,
            "user_name": user["name"],
            "recommendations": recommendations
        }


# ======================================================
# ENGINE INSTANCE
# ======================================================
recommendation_engine = RecommendationEngine()


# ======================================================
# API ENDPOINTS
# ======================================================
@app.post("/users/")
def register_user(user: User):
    # Basic eligibility rules
    if not (21 <= user.age <= 24):
        return {"eligible": False, "reason": "Age must be between 21 and 24 years"}
    if user.family_annual_income > 800000:
        return {"eligible": False, "reason": "Family annual income must be ≤ 8 lakhs"}

    user_id = recommendation_engine.register_user(user.dict())
    return {
        "user_id": user_id,
        "eligible": True,
        "reason": "Eligible",
        "message": "Registered"
    }


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, limit: int = 5):
    try:
        return recommendation_engine.get_recommendations(user_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
