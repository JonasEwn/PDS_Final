import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessing_JSON_Seniority:
    """
    Preprocess LinkedIn-style JSON data for seniority prediction.

    Each outer list element = one person
    Uses the most recent ACTIVE job only
    text  = position
    label = seniority

    Output:
        self.X       -> pd.Series (text)
        self.y       -> np.ndarray (encoded labels)
        self.y_str   -> pd.Series (string labels)
        self.df      -> full DataFrame
    """

    def __init__(
        self,
        json_path: str,
        label_encoder: LabelEncoder,
        drop_label: str = "Professional"
    ):
        self.json_path = json_path
        self.drop_label = drop_label
        self.label_encoder = label_encoder  # IMPORTANT: reuse from train

        self.df = None
        self.X = None
        self.y = None
        self.y_str = None

        self.read_json()

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _parse_year_month(s):
        if not isinstance(s, str) or len(s) < 7:
            return None
        try:
            y, m = s.split("-")
            return int(y), int(m)
        except Exception:
            return None

    @staticmethod
    def clean_text(text: str) -> str:
        return str(text).lower().strip()

    # ----------------------------
    # Core logic
    # ----------------------------
    def read_json(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)

        rows = []

        for person in data:
            active_jobs = []

            for job in person:
                if job.get("status") != "ACTIVE":
                    continue

                start = self._parse_year_month(job.get("startDate"))
                if start is None:
                    continue

                active_jobs.append((start, job))

            if not active_jobs:
                continue

            # Most recent ACTIVE job
            active_jobs.sort(key=lambda x: x[0])
            job = active_jobs[-1][1]

            position = job.get("position")
            seniority = job.get("seniority")

            if not position or not seniority:
                continue

            if seniority == self.drop_label:
                continue

            rows.append({
                "text": self.clean_text(position),
                "label": seniority
            })

        self.df = pd.DataFrame(rows)

        # === Match CSV class output ===
        self.X = self.df["text"]
        self.y_str = self.df["label"]

        # IMPORTANT: do NOT fit again on test data
        self.y = self.label_encoder.transform(self.y_str)

        print(f"[JSON] Loaded {len(self.df)} samples")
