import pandas as pd
import json
import math

class PreprocessingDomain:
    """
    Domain/Department preprocessing

    Key ideas:
    - Target job = ACTIVE job with most recent startDate
    - Cutoff = target startDate  -> features only from jobs before cutoff (anti-leakage)
    - Numeric features that can correlate with department patterns:
      * stability/volatility, employer concentration, internal mobility proxies,
        active job patterns, org-repeat ratios, overlap of active roles, etc.
    """

    X = None
    Y = None
    data = None

    def __init__(self, X_link, department_link, seniority_link=None, label_col="Department (Label 2)"):
        self.label_col = label_col
        self.annotated_data = self.prepare_json(X_link)
        self.diff_departments = self.prepare_csv(department_link)
        self.diff_seniorities = self.prepare_csv(seniority_link) if seniority_link else None

        self.clean_annotated_data()
        self.init_department_encoding_from_json()


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
        if a is None or b is None:
            return False
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


    def _ym_le(self, a, b):
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


    def init_department_encoding_from_json(self):
        found = set()

        for person in self.annotated_data:
            for job in person:
                d = self._norm(job.get("department"))
                if d:
                    found.add(d)

        # Encoding labels from json file
        self.department_categories = sorted(found)
        self.department_to_id = {lab: i for i, lab in enumerate(self.department_categories)}
        self.id_to_department = {i: lab for lab, i in self.department_to_id.items()}

        print(f"{len(self.department_categories)} departments found:")
        print(self.department_categories)


    def get_target_active_job(self, person):
        """
        Target/current job = ACTIVE job with most recent startDate.
        """
        active = []
        for job in person:
            if job.get("status") == "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                if start is not None:
                    active.append((start, job))
        if not active:
            return None
        active.sort(key=lambda x: x[0])
        return active[-1][1]


    def current_department_id(self, person):
        job = self.get_target_active_job(person)
        if job is None:
            raise ValueError("No ACTIVE job found for this person.")

        d = self._norm(job.get("department"))
        if not d:
            raise ValueError("ACTIVE job has no 'department' value.")

        if d not in self.department_to_id:
            raise KeyError(f"Unknown department '{d}' not in encoding.")

        return self.department_to_id[d]


    def history_before_target(self, person):
        """
        Jobs with startDate strictly before target startDate.
        """
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

        return hist, t0, target


    def num_of_jobs(self, jobs):
        return len(jobs)


    def num_employers(self, jobs):
        orgs = set()
        for job in jobs:
            org = self._norm(job.get("organization"))
            if org:
                orgs.add(org)
        return len(orgs)


    def career_length(self, history, cutoff_ym):
        if not history:
            return 0
        starts = [self._parse_year_month(j.get("startDate")) for j in history]
        starts = [s for s in starts if s is not None]
        if not starts:
            return 0
        career_start = min(starts)
        return self._months_between(career_start, cutoff_ym) or 0


    def job_duration_clipped(self, job, cutoff_ym):
        start = self._parse_year_month(job.get("startDate"))
        end = self._parse_year_month(job.get("endDate"))
        if start is None:
            return None
        if end is None or not self._ym_le(end, cutoff_ym):
            end = cutoff_ym
        return self._months_between(start, end)


    def avg_job_duration(self, history, cutoff_ym):
        if not history:
            return 0
        lengths = []
        for job in history:
            length = self.job_duration_clipped(job, cutoff_ym)
            if length is not None:
                lengths.append(length)
        return (sum(lengths) / len(lengths)) if lengths else 0


    def longest_job_duration(self, history, cutoff_ym):
        if not history:
            return 0
        lengths = []
        for job in history:
            length = self.job_duration_clipped(job, cutoff_ym)
            if length is not None:
                lengths.append(length)
        return max(lengths) if lengths else 0


    def std_job_duration(self, history, cutoff_ym):
        if not history:
            return 0
        lengths = []
        for job in history:
            length = self.job_duration_clipped(job, cutoff_ym)
            if length is not None:
                lengths.append(length)

        n = len(lengths)
        if n < 2:
            return 0
        mean = sum(lengths) / n
        var = sum((x - mean) ** 2 for x in lengths) / (n - 1)
        return math.sqrt(var)


    def time_since_last_change(self, history, cutoff_ym):
        if not history:
            return 0
        changes = []
        for job in history:
            s = self._parse_year_month(job.get("startDate"))
            e = self._parse_year_month(job.get("endDate"))
            if s is not None:
                changes.append(s)
            if e is not None and self._ym_le(e, cutoff_ym):
                changes.append(e)
        if not changes:
            return 0
        last_change = max(changes)
        return self._months_between(last_change, cutoff_ym) or 0


    def num_active_jobs(self, person):
        return sum(1 for job in person if job.get("status") == "ACTIVE")


    def active_jobs_same_org_ratio(self, person):
        """
        Ratio of ACTIVE jobs that share the most common organization among ACTIVE jobs.
        Captures cases like: two parallel ACTIVE roles at the same company (CFO + Prokurist).
        """
        active = [job for job in person if job.get("status") == "ACTIVE"]
        if not active:
            return 0
        orgs = [self._norm(j.get("organization")) for j in active if self._norm(j.get("organization"))]
        if not orgs:
            return 0
        # most common org count
        from collections import Counter
        c = Counter(orgs)
        most_common = c.most_common(1)[0][1]
        return most_common / len(active)


    def prev_job_same_org_as_target(self, target, history):
        if not history:
            return 0
        target_org = self._norm(target.get("organization"))
        if not target_org:
            return 0
        prev = []
        for j in history:
            start = self._parse_year_month(j.get("startDate"))
            if start is not None:
                prev.append((start, j))
        if not prev:
            return 0
        prev.sort(key=lambda x: x[0])
        last_prev = prev[-1][1]
        return int(self._norm(last_prev.get("organization")) == target_org)


    def num_prev_jobs_same_org_as_target(self, target, history):
        target_org = self._norm(target.get("organization"))
        if not target_org:
            return 0
        return sum(1 for j in history if self._norm(j.get("organization")) == target_org)


    def org_repeat_ratio(self, target, history):
        denom = max(1, self.num_of_jobs(history))
        return self.num_prev_jobs_same_org_as_target(target, history) / denom


    def employer_concentration(self, history):
        """
        max jobs at any single org / total jobs
        High for people who stick in one org (often ops/admin), lower for consulting etc.
        """
        if not history:
            return 0
        from collections import Counter
        orgs = [self._norm(j.get("organization")) for j in history if self._norm(j.get("organization"))]
        if not orgs:
            return 0
        c = Counter(orgs)
        return max(c.values()) / max(1, len(history))


    def stability_score(self, history, cutoff_ym):
        cl = max(1, self.career_length(history, cutoff_ym))
        return self.longest_job_duration(history, cutoff_ym) / cl


    def volatility_score(self, history, cutoff_ym):
        avg = max(1e-6, self.avg_job_duration(history, cutoff_ym))
        return self.std_job_duration(history, cutoff_ym) / avg


    def job_switch_rate(self, history, cutoff_ym):
        months = self.career_length(history, cutoff_ym)
        if months <= 0:
            return 0
        years = max(months / 12.0, 1.0)
        return self.num_of_jobs(history) / years


    def internal_mobility_proxy(self, history):
        """
        Fraction of job-to-job transitions where employer stays the same.
        (A proxy for internal moves; can correlate with some departments.)
        """
        if len(history) < 2:
            return 0
        seq = []
        for j in history:
            s = self._parse_year_month(j.get("startDate"))
            org = self._norm(j.get("organization"))
            if s and org:
                seq.append((s, org))
        if len(seq) < 2:
            return 0
        seq.sort(key=lambda x: x[0])
        same = 0
        total = 0
        for i in range(1, len(seq)):
            total += 1
            if seq[i][1] == seq[i-1][1]:
                same += 1
        return same / total if total else 0


    def data_pipeline(self):
        rows = []
        dropped_no_active_or_start = 0
        dropped_missing_label = 0

        for p_id, person in enumerate(self.annotated_data):
            history, cutoff_ym, target = self.history_before_target(person)
            if target is None or cutoff_ym is None:
                dropped_no_active_or_start += 1
                continue

            # label = department of target ACTIVE job
            try:
                label = self.current_department_id(person)
            except ValueError:
                dropped_missing_label += 1
                continue

            row = {
                "Department (Label 2)": label,

                # Old features
                "num_prev_jobs": self.num_of_jobs(history),
                "career_length_months": self.career_length(history, cutoff_ym),
                "avg_prev_job_duration": self.avg_job_duration(history, cutoff_ym),
                "time_since_last_change": self.time_since_last_change(history, cutoff_ym),
                "longest_prev_job_duration": self.longest_job_duration(history, cutoff_ym),
                "std_prev_job_duration": self.std_job_duration(history, cutoff_ym),
                "num_prev_employers": self.num_employers(history),
                "job_switch_rate_per_year": self.job_switch_rate(history, cutoff_ym),

                # New -> For Domain
                "num_active_jobs": self.num_active_jobs(person),
                "active_jobs_same_org_ratio": self.active_jobs_same_org_ratio(person),
                "prev_job_same_org_as_target": self.prev_job_same_org_as_target(target, history),
                "num_prev_jobs_same_org_as_target": self.num_prev_jobs_same_org_as_target(target, history),
                "org_repeat_ratio": self.org_repeat_ratio(target, history),
                "employer_concentration": self.employer_concentration(history),
                "stability_score": self.stability_score(history, cutoff_ym),
                "volatility_score": self.volatility_score(history, cutoff_ym),
                "internal_mobility_proxy": self.internal_mobility_proxy(history),
            }

            rows.append(row)

        self.data = pd.DataFrame(rows)

        self.Y = self.data[self.label_col]
        self.X = self.data.drop(columns=[self.label_col])

        print(f"Dropped (no ACTIVE job or missing startDate): {dropped_no_active_or_start}")
        print(f"Dropped (ACTIVE job missing department label): {dropped_missing_label}")
        return self.X, self.Y
