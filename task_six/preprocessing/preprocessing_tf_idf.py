import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class PreprocessingTFIDF:
    X = None            # TF-IDF sparse matrix
    Y = None            # encoded labels
    data = None         # dataframe with text + label_id
    X_text = None       # list of raw texts (helpful for debugging)

    def __init__(
        self,
        annotated_json_link,
        seniority_label_list_csv=None,  # optional: your "text,label" file
        label_col="seniority",
        include_history=True,
        include_org=True,
        include_dates=False,            # usually False for BoW baseline
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000
    ):
        self.label_col = label_col
        self.include_history = include_history
        self.include_org = include_org
        self.include_dates = include_dates

        self.annotated_data = self._prepare_json(annotated_json_link)

        # optional mapping file (text,label) -> used for normalizing titles
        self.title_to_label = None
        if seniority_label_list_csv is not None:
            df_map = pd.read_csv(seniority_label_list_csv)
            # normalize keys
            self.title_to_label = {
                self._norm(row["text"]): self._norm(row["label"])
                for _, row in df_map.iterrows()
                if isinstance(row.get("text"), str) and isinstance(row.get("label"), str)
            }

        self._clean_annotated_data()
        self._init_label_encoding_from_json()

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            lowercase=True
        )

        # build X_text, Y, X
        self.data_pipeline()

    # ---------- IO ----------
    def _prepare_json(self, link):
        with open(link, "r") as f:
            return json.load(f)

    # ---------- utils ----------
    def _norm(self, v):
        if v is None:
            return ""
        return str(v).strip().lower()

    def _parse_year_month(self, s):
        if s is None or not isinstance(s, str) or len(s) < 7:
            return None
        try:
            y, m = s.split("-")
            y, m = int(y), int(m)
            if not (1 <= m <= 12):
                return None
            return (y, m)
        except Exception:
            return None

    def _ym_lt(self, a, b):
        if a is None or b is None:
            return False
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    # ---------- cleaning ----------
    def _clean_annotated_data(self):
        cleaned = []
        dup = 0
        for person in self.annotated_data:
            seen = set()
            unique = []
            for job in person:
                key = (
                    self._norm(job.get("organization")),
                    self._norm(job.get("position")),
                    self._norm(job.get("startDate")),
                    self._norm(job.get("endDate")),
                    self._norm(job.get("status")),
                    self._norm(job.get("seniority")),
                )
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)
                unique.append(job)
            cleaned.append(unique)
        self.annotated_data = cleaned
        print(f"{dup} duplicates removed")

    # ---------- target definition ----------
    def get_target_active_job(self, person):
        active = []
        for job in person:
            if job.get("status") == "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                if start is not None:
                    active.append((start, job))
        if not active:
            return None
        active.sort(key=lambda x: x[0])
        return active[-1][1]  # most recent ACTIVE

    def history_before_target(self, person):
        target = self.get_target_active_job(person)
        if target is None:
            return None, None, None
        t0 = self._parse_year_month(target.get("startDate"))
        if t0 is None:
            return None, None, None

        hist = []
        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            if start is None:
                continue
            if self._ym_lt(start, t0):
                hist.append(job)
        return target, hist, t0

    # ---------- labels ----------
    def _init_label_encoding_from_json(self):
        found = set()
        for person in self.annotated_data:
            for job in person:
                s = self._norm(job.get("seniority"))
                if s:
                    found.add(s)

        # stable ordering (you can hardcode order if you want)
        self.label_categories = sorted(found)
        self.label_to_id = {lab: i for i, lab in enumerate(self.label_categories)}
        self.id_to_label = {i: lab for lab, i in self.label_to_id.items()}

        print(f"{len(self.label_categories)} seniority labels found:")
        print(self.label_categories)

    def _label_id_for_person(self, person):
        target = self.get_target_active_job(person)
        if target is None:
            raise ValueError("No ACTIVE job found")
        s = self._norm(target.get("seniority"))
        if not s:
            raise ValueError("ACTIVE job has no seniority label")
        if s not in self.label_to_id:
            raise KeyError(f"Unknown label: {s}")
        return self.label_to_id[s]

    # ---------- text building ----------
    def _normalize_title_via_mapping(self, title):
        """Optional: map raw title -> canonical label name (if present in csv)."""
        if self.title_to_label is None:
            return title
        key = self._norm(title)
        mapped = self.title_to_label.get(key)
        return mapped if mapped else title

    def _job_to_text(self, job, role=""):
        parts = []
        pos = job.get("position")
        org = job.get("organization")

        if pos:
            pos_txt = self._normalize_title_via_mapping(pos)
            parts.append(str(pos_txt))

        if self.include_org and org:
            parts.append(str(org))

        if self.include_dates:
            sd = job.get("startDate")
            ed = job.get("endDate")
            if sd:
                parts.append(f"start_{sd}")
            if ed:
                parts.append(f"end_{ed}")

        if role:
            parts.append(role)

        return " ".join(parts)

    def build_person_text(self, person):
        target, hist, cutoff = self.history_before_target(person)
        if target is None:
            return None

        # current job text
        text_parts = [self._job_to_text(target, role="current")]

        # add history
        if self.include_history and hist:
            # sort by start date for consistent ordering
            hist_sorted = []
            for j in hist:
                st = self._parse_year_month(j.get("startDate"))
                if st is not None:
                    hist_sorted.append((st, j))
            hist_sorted.sort(key=lambda x: x[0])

            for _, j in hist_sorted:
                text_parts.append(self._job_to_text(j, role="past"))

        return " ".join(text_parts)

    # ---------- pipeline ----------
    def data_pipeline(self):
        rows = []
        dropped_no_active = 0
        dropped_missing_label = 0

        for person in self.annotated_data:
            txt = self.build_person_text(person)
            if txt is None:
                dropped_no_active += 1
                continue

            try:
                y = self._label_id_for_person(person)
            except ValueError:
                dropped_missing_label += 1
                continue

            rows.append({"text": txt, "label_id": y})

        self.data = pd.DataFrame(rows)
        self.X_text = self.data["text"].tolist()
        self.Y = self.data["label_id"]

        # TF-IDF fit on training set (here: full data; do split outside for clean eval)
        self.X = self.vectorizer.fit_transform(self.X_text)

        print(f"Dropped (no ACTIVE job or missing startDate): {dropped_no_active}")
        print(f"Dropped (missing seniority label): {dropped_missing_label}")
        print("X shape:", self.X.shape)
