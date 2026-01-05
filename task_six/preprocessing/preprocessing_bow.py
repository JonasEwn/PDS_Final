import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class PreprocessingBOW:
    """
    Bag-of-Words preprocessing for Seniority classification.

    Output:
      - self.X_text : list[str] raw constructed texts (debugging)
      - self.X      : sparse BoW matrix (CountVectorizer)
      - self.Y      : encoded labels (int)
      - self.data   : DataFrame with "text" and label column (encoded)
    """

    X = None
    Y = None
    data = None
    X_text = None

    def __init__(
        self,
        annotated_json_link: str,
        seniority_label_list_csv: str | None = None,  # optional (see _normalize_title_via_mapping)
        label_col: str = "Seniority (Label 1)",
        include_history: bool = True,
        include_org: bool = True,
        include_dates: bool = False,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float | int = 0.95,
        max_features: int = 50_000,
        merge_labels: dict[str, str] | None = None,    # e.g. {"director":"management","junior":"professional"}
    ):
        self.label_col = label_col
        self.include_history = include_history
        self.include_org = include_org
        self.include_dates = include_dates
        self.merge_labels = {self._norm(k): self._norm(v) for k, v in (merge_labels or {}).items()}

        self.annotated_data = self._prepare_json(annotated_json_link)

        # optional mapping file (text,label) - used to normalize titles (optional, can be risky)
        self.title_to_label = None
        if seniority_label_list_csv is not None:
            df_map = pd.read_csv(seniority_label_list_csv)
            # safe normalization of dict keys/values
            self.title_to_label = {
                self._norm(row["text"]): self._norm(row["label"])
                for _, row in df_map.iterrows()
                if isinstance(row.get("text"), str) and isinstance(row.get("label"), str)
            }

        self._clean_annotated_data()
        self._init_label_encoding_from_json()

        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            lowercase=True
        )

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
    def _apply_merge(self, label_norm: str) -> str:
        """Merge extreme minority labels if configured."""
        return self.merge_labels.get(label_norm, label_norm)

    def _init_label_encoding_from_json(self):
        found = set()
        for person in self.annotated_data:
            for job in person:
                s = self._norm(job.get("seniority"))
                if s:
                    found.add(self._apply_merge(s))

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

        s = self._apply_merge(s)

        if s not in self.label_to_id:
            raise KeyError(f"Unknown label after merging: {s}")
        return self.label_to_id[s]

    # ---------- text building ----------
    def _normalize_title_via_mapping(self, title: str) -> str:
        """
        Optional: map exact title text -> canonical token.
        NOTE: This can hurt if your csv 'label' are broad ("junior") rather than title variants.
        """
        if self.title_to_label is None:
            return title
        mapped = self.title_to_label.get(self._norm(title))
        return mapped if mapped else title

    def _job_to_text(self, job, role=""):
        parts = []
        pos = job.get("position")
        org = job.get("organization")

        if pos:
            parts.append(str(self._normalize_title_via_mapping(pos)))

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

        text_parts = [self._job_to_text(target, role="current")]

        if self.include_history and hist:
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
        dropped_no_active_or_start = 0
        dropped_missing_label = 0

        for person in self.annotated_data:
            txt = self.build_person_text(person)
            if txt is None:
                dropped_no_active_or_start += 1
                continue

            try:
                y = self._label_id_for_person(person)
            except ValueError:
                dropped_missing_label += 1
                continue

            rows.append({self.label_col: y, "text": txt})

        self.data = pd.DataFrame(rows)
        self.X_text = self.data["text"].tolist()
        self.Y = self.data[self.label_col]
        self.X = self.vectorizer.fit_transform(self.X_text)

        print(f"Dropped (no ACTIVE job or missing startDate): {dropped_no_active_or_start}")
        print(f"Dropped (missing seniority label): {dropped_missing_label}")
        print("X shape:", self.X.shape)
