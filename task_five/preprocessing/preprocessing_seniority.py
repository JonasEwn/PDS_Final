import pandas as pd
import json
import math

class Preprocessing:
    X = None
    Y = None
    data = None

    def __init__(self, X_link, department_link, seniority_link, label_col="Seniority (Label 1)"):
        self.label_col = label_col
        self.annotated_data = self.prepare_json(X_link)
        self.diff_departments = self.prepare_csv(department_link)
        self.diff_seniorities = self.prepare_csv(seniority_link)

        self.seniorities = {
            "junior": 0,
            "professional": 1,
            "senior": 2,
            "lead": 3,
            "management": 4,
            "director": 5,
        }

        self.clean_annotated_data()
        self.init_seniority_encoding_from_json()


    def prepare_csv(self, link):
        return pd.read_csv(link)


    def prepare_json(self, link):
        with open(link, "r") as p:
            return json.load(p)


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


    def _months_between(self, start, end):
        if start is None or end is None:
            return None
        (y1, m1), (y2, m2) = start, end
        return max(0, (y2 - y1) * 12 + (m2 - m1))


    def _ym_lt(self, a, b):
        """
        True if a < b for (year, month).
        """
        if a is None or b is None:
            return False
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


    def _ym_le(self, a, b):
        """
        True if a <= b for (year, month).
        """
        if a is None or b is None:
            return False
        return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])


    def clean_annotated_data(self):
        cleaned_data = []
        duplicates_found = 0

        for person in self.annotated_data:
            seen = set()
            unique_jobs = []
            for job in person:
                key = (
                    self._norm(job.get("organization")),
                    self._norm(job.get("position")),
                    self._norm(job.get("startDate")),
                    self._norm(job.get("endDate")),
                    self._norm(job.get("status")),
                    self._norm(job.get("department")),
                    self._norm(job.get("seniority")),
                )
                if key not in seen:
                    seen.add(key)
                    unique_jobs.append(job)
                else:
                    duplicates_found += 1
            cleaned_data.append(unique_jobs)

        print(f"{duplicates_found} duplicates have been removed")
        self.annotated_data = cleaned_data


    def init_seniority_encoding_from_json(self):
        found = set()
        for person in self.annotated_data:
            for job in person:
                s = job.get("seniority")
                s_norm = self._norm(s)
                if s_norm:
                    found.add(s_norm)

        ordered = [lab for lab in self.seniorities.keys() if lab in found]
        extras = sorted(found - set(self.seniorities.keys()))
        if extras:
            raise ValueError(f"Unexpected seniorities in JSON: {sorted(extras)}")

        self.seniority_categories = ordered
        self.seniority_to_id = {lab: i for i, lab in enumerate(self.seniority_categories)}
        self.id_to_seniority = {i: lab for lab, i in self.seniority_to_id.items()}

        print(f"{len(self.seniority_categories)} seniorities found:")
        print(self.seniority_categories)


    def get_target_active_job(self, person):
        active = []
        for job in person:
            if job.get("status") == "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                if start is not None:
                    active.append((start, job))

        if not active: return None

        # most recent start date
        active.sort(key=lambda x: x[0])
        return active[-1][1]


    def current_seniority_id(self, person):
        job = self.get_target_active_job(person)
        if job is None:
            raise ValueError("No ACTIVE job found for this person.")

        s = job.get("seniority")
        s_norm = self._norm(s)
        if not s_norm:
            raise ValueError("ACTIVE job has no 'seniority' value.")

        if s_norm not in self.seniority_to_id:
            raise KeyError(f"Unknown seniority '{s}' (normalized='{s_norm}') not in encoding.")

        return self.seniority_to_id[s_norm]


    def _target_start(self, person):
        job = self.get_target_active_job(person)
        if job is None:
            return None
        return self._parse_year_month(job.get("startDate"))


    def history_before_target(self, person):
        t0 = self._target_start(person)
        if t0 is None:
            return None, None  # no ACTIVE job or missing start

        hist = []
        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            if start is None:
                continue
            if self._ym_lt(start, t0):
                hist.append(job)

        return hist, t0


    def num_of_jobs(self, jobs_of_person):
        return len(jobs_of_person)


    def career_length(self, history, cutoff_ym):
        if not history:
            return 0

        starts = [self._parse_year_month(j.get("startDate")) for j in history]
        starts = [s for s in starts if s is not None]
        if not starts:
            return 0

        career_start = min(starts)
        return self._months_between(career_start, cutoff_ym) or 0


    def avg_job_duration(self, history, cutoff_ym):
        if not history: return 0

        lengths = []
        for job in history:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start is None:
                continue

            # if missing end -> clip at cutoff
            if end is None or not self._ym_le(end, cutoff_ym):
                end = cutoff_ym

            length = self._months_between(start, end)
            if length is not None:
                lengths.append(length)

        return (sum(lengths) / len(lengths)) if lengths else 0


    def time_since_last_job_change(self, history, cutoff_ym):
        if not history: return 0

        changes = []
        for job in history:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))
            if start is not None:
                changes.append(start)
            if end is not None and self._ym_le(end, cutoff_ym):
                changes.append(end)

        if not changes:
            return 0

        last_change = max(changes)
        return self._months_between(last_change, cutoff_ym) or 0


    def longest_job_duration(self, history, cutoff_ym):
        if not history: return 0

        lengths = []
        for job in history:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start is None:
                continue

            if end is None or not self._ym_le(end, cutoff_ym):
                end = cutoff_ym

            length = self._months_between(start, end)
            if length is not None:
                lengths.append(length)

        return max(lengths) if lengths else 0


    def std_job_duration(self, history, cutoff_ym):
        if not history: return 0

        lengths = []
        for job in history:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start is None:
                continue
            if end is None or not self._ym_le(end, cutoff_ym):
                end = cutoff_ym

            length = self._months_between(start, end)
            if length is not None:
                lengths.append(length)

        n = len(lengths)
        if n < 2:
            return 0

        mean = sum(lengths) / n
        var = sum((x - mean) ** 2 for x in lengths) / (n - 1)  # sample variance
        return math.sqrt(var)


    def num_employers(self, history):
        orgs = set()
        for job in history:
            org = self._norm(job.get("organization"))
            if org:
                orgs.add(org)
        return len(orgs)


    def job_switch_rate(self, history, cutoff_ym):
        months = self.career_length(history, cutoff_ym)
        if months <= 0:
            return 0
        years = months / 12.0
        years = max(years, 1.0)  # stabilize for very short histories
        return self.num_of_jobs(history) / years


    def avg_jobs_per_employer(self, history):
        n_emp = self.num_employers(history)
        if n_emp == 0:
            return 0
        return self.num_of_jobs(history) / n_emp


    def num_prev_jobs_same_org_as_target(self, person, history):
        target = self.get_target_active_job(person)
        if target is None: return 0

        target_org = self._norm(target.get("organization"))
        if not target_org: return 0

        return sum(1 for j in history if self._norm(j.get("organization")) == target_org)


    def prev_job_same_org_as_target(self, person, history):
        target = self.get_target_active_job(person)
        if target is None:
            return 0
        target_org = self._norm(target.get("organization"))
        if not target_org or not history:
            return 0

        prev = []
        for j in history:
            start = self._parse_year_month(j.get("startDate"))
            if start is not None:
                prev.append((start, j))
        if not prev:
            return 0

        prev.sort(key=lambda x: x[0])
        last_prev_job = prev[-1][1]
        last_prev_org = self._norm(last_prev_job.get("organization"))
        return int(last_prev_org == target_org)


    def num_active_jobs(self, person):
        return sum(1 for job in person if job.get("status") == "ACTIVE")


    def data_pipeline(self):
        rows = []
        dropped_no_active_or_start = 0
        dropped_missing_label = 0

        for p_id, person in enumerate(self.annotated_data):
            target = self.get_target_active_job(person)
            if target is None:
                dropped_no_active_or_start += 1
                continue

            cutoff = self._parse_year_month(target.get("startDate"))
            if cutoff is None:
                dropped_no_active_or_start += 1
                continue

            try:
                label = self.current_seniority_id(person)
            except ValueError:
                dropped_missing_label += 1
                continue

            history, cutoff_ym = self.history_before_target(person)
            if history is None:
                dropped_no_active_or_start += 1
                continue

            row = {
                "Seniority (Label 1)": label,
                "num_prev_jobs": self.num_of_jobs(history),
                "career_length_months": self.career_length(history, cutoff_ym),
                "avg_prev_job_duration": self.avg_job_duration(history, cutoff_ym),
                "time_since_last_change": self.time_since_last_job_change(history, cutoff_ym),
                "longest_prev_job_duration": self.longest_job_duration(history, cutoff_ym),
                "std_prev_job_duration": self.std_job_duration(history, cutoff_ym),
                "num_prev_employers": self.num_employers(history),
                "job_switch_rate_per_year": self.job_switch_rate(history, cutoff_ym),
                "avg_jobs_per_employer": self.avg_jobs_per_employer(history),
                "prev_job_same_org_as_target": self.prev_job_same_org_as_target(person, history),
                "num_prev_jobs_same_org_as_target": self.num_prev_jobs_same_org_as_target(person, history),
                "num_active_jobs": self.num_active_jobs(person),
            }

            rows.append(row)

        self.data = pd.DataFrame(rows)
        self.Y = self.data[self.label_col]
        self.X = self.data.drop(columns=[self.label_col])

        print(f"Dropped (no ACTIVE job or missing startDate): {dropped_no_active_or_start}")
        print(f"Dropped (ACTIVE job missing seniority label): {dropped_missing_label}")
