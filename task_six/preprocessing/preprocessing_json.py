import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class Preprocessing_JSON_annotated_Seniority():
    def __init__(self, path):
        self.path = path
        self.labels = [
            "Junior",
            "Professional",
            "Senior",
            "Lead",
            "Management",
            "Director"
        ]
        self.label_encoder = OrdinalEncoder(
            categories=[self.labels],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.df = None
        self.X = None
        self.y = None

        self.read_json()


    @staticmethod
    def _parse_year_month(s):
        """
        Expects: "YYYY.MM" -> Returns: (year, month)
        """
        if not isinstance(s, str) or len(s) < 7: return None

        try:
            year, month = s.split("-")
            return int(year), int(month)
        except Exception: return None


    @staticmethod
    def clean_text(text):
        return str(text).lower().strip().replace("-", " ").replace("/", " ")


    def read_json(self):
        with open(self.path, "r") as f: data = json.load(f)

        rows = []

        for person in data:
            active_jobs = []

            for job in person:
                # Only active jobs
                if job.get("status") != "ACTIVE": continue

                start = self._parse_year_month(job.get("startDate"))
                if start is None: continue

                active_jobs.append((start, job))

            if not active_jobs: continue

            _, job = max(active_jobs, key=lambda x: x[0])

            position = job.get("position")
            seniority = job.get("seniority")

            if not position or not seniority: continue

            rows.append({
                "text": self.clean_text(position),
                "label": seniority
            })

        self.df = pd.DataFrame(rows)

        if self.df.empty:
            raise ValueError("No valid samples found in JSON")

        self.X = self.df["text"]

        self.y = self.label_encoder.fit_transform(
            self.df["label"].values.reshape(-1, 1)
        ).flatten()

        print(
            f"[JSON] Loaded {len(self.df)}"
        )