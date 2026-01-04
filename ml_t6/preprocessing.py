import pandas as pd
import json
import math
import re

class Preprocessing():
    X = None
    Y = None
    data = None


    def __init__(self, X_link, department_link, seniority_link, label_col = "Seniority (Label 1)"):
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

        # Clean Data in the same step
        self.clean_annotated_data()

        # Label Encoding
        self.init_seniority_encoding_from_json()



    def prepare_csv(self, link):
        """
        ...
        """
        df = pd.read_csv(link)
        #print(df)

        #seniorities = df["label"].dropna().unique()
        #print(f"{len(seniorities)} Seniorities have been found.\nSeniorities: {seniorities}")

        return df


    def init_seniority_encoding_from_json(self):
        found = set()

        for person in self.annotated_data:
            for job in person:
                s = job.get("seniority")
                if s is None:
                    continue
                s_norm = self._norm(s)
                if s_norm:
                    found.add(s_norm)

        # Use your predefined order, but only keep labels that actually exist in the JSON
        ordered = [lab for lab in self.seniorities.keys() if lab in found]

        # If there are unexpected labels in the JSON, append them at the end (stable, sorted)
        extras = sorted(found - set(self.seniorities.keys()))
        if extras:
            raise ValueError(f"Unexpected seniorities in JSON: {sorted(extras)}")
        self.seniority_categories = ordered + extras

        # Build encoding using the intended order (and then extras)
        self.seniority_to_id = {lab: i for i, lab in enumerate(self.seniority_categories)}
        self.id_to_seniority = {i: lab for lab, i in self.seniority_to_id.items()}

        print(f"{len(self.seniority_categories)} seniorities found:")
        print(self.seniority_categories)


    def get_main_active_job(self, person, todays_date="2026-01"):
        ref = self._parse_year_month(todays_date)

        best_job = None
        best_duration = -1

        for job in person:
            if job.get("status") == "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                if start:
                    dur = self._months_between(start, ref)
                    if dur is not None and dur > best_duration:
                        best_duration = dur
                        best_job = job
        return best_job


    def current_seniority_id(self, person, todays_date="2026-01"):
        job = self.get_main_active_job(person, todays_date=todays_date)

        if job is None:
            raise ValueError("No ACTIVE job found for this person.")

        s = job.get("seniority")
        if s is None or not self._norm(s):
            raise ValueError("ACTIVE job has no 'seniority' value.")

        s_norm = self._norm(s)

        if s_norm not in self.seniority_to_id:
            raise KeyError(f"Unknown seniority '{s}' (normalized='{s_norm}') not in encoding.")

        return self.seniority_to_id[s_norm]


    def prepare_json(self, link):
        """
        Iterates over each
        """
        with open(link, "r") as p:
            data = json.load(p)
        return data


    def _norm(self, v):
        """
        Remove Uppercase to further detect duplicates
        """
        if v is None:
            return ""
        return str(v).strip().lower()


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


    def num_of_jobs(self, jobs_of_person):
        return len(jobs_of_person)


    def _months_between(self, start, end):
        if start is None or end is None: return None
        (y1, m1), (y2, m2) = start, end
        return max(0, (y2 - y1) * 12 + (m2 - m1))


    def _parse_year_month(self, s):
        if s is None or not isinstance(s, str) or len(s) < 7:
            return None
        try:
            y, m = s.split("-")
            return int(y), int(m)
        except:
            return None


    def career_length(self, person, todays_date="2026-01"):
        todays_date = self._parse_year_month(todays_date)

        start_dates = []
        end_dates = []
        is_active = False

        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start: start_dates.append(start)
            if end: end_dates.append(end)

            if job.get("status") == "ACTIVE": is_active = True

        # Wenn es kein start Datum gibt gebe einfach 0 zurück
        if not start_dates: return 0

        career_start = min(start_dates) #Min value of start Dates as our actual start

        if is_active:
            career_end = todays_date
        else:
            if end_dates:
                career_end = max(end_dates)
            else:
                career_end = career_start

        return self._months_between(career_start, career_end)


    def is_active(self, person):
        #is_active = 0
        #for job in person:
        #    is_active = 1 if any(job["status"] == "ACTIVE") else 0
        #return is_active
        return int(any(job.get("status") == "ACTIVE" for job in person))


    def avg_job_duration(self, person, todays_date="2026-01"):
        todays_date = self._parse_year_month(todays_date)

        job_lengths = []

        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start is None: continue

            # If job is still active
            if end is None and job.get("status") == "ACTIVE": end = todays_date

            length = self._months_between(start, end)

            if length is not None:
                job_lengths.append(length)

        if not job_lengths:
            return 0
        return sum(job_lengths) / len(job_lengths)


    def current_job_duration(self, person, todays_date="2026-01"):
        """
        Duration of current Job

        0 means Inactive
        If 2 Jobs returns the longest -> Is accounted for in num_jobs_active()
        """
        todays_date = self._parse_year_month(todays_date)
        durations = []

        for job in person:
            if job.get("status") == "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                if start:
                    durations.append(self._months_between(start, todays_date))

        #print(f"Durations {durations}")
        if not durations: return 0
        return max(durations)


    def num_active_jobs(self, person):
        return sum(1 for job in person if job.get("status") == "ACTIVE")


    def time_since_job_change(self, person, todays_date="2026-01"):
        todays_date = self._parse_year_month(todays_date)
        job_changes = []

        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start: job_changes.append(start)
            if end: job_changes.append(end)

        if not job_changes: return 0
        last_change = max(job_changes)
        return self._months_between(last_change, todays_date)


    def longest_job_duration(self, person, todays_date="2026-01"):
        todays_date = self._parse_year_month(todays_date)
        lengths = []

        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start is None: continue

            # If ACTIVE end is today
            if end is None and job.get("status") == "ACTIVE": end = todays_date

            length = self._months_between(start, end)
            if length is not None:
                lengths.append(length)

        if not lengths:
            return 0
        return max(lengths)


    def std_job_duration(self, person, todays_date="2026-01"):
        todays_date = self._parse_year_month(todays_date)
        lengths = []

        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            end = self._parse_year_month(job.get("endDate"))

            if start is None: continue
            if end is None and job.get("status") == "ACTIVE": end = todays_date

            length = self._months_between(start, end)
            if length is not None:
                lengths.append(length)

        if len(lengths) < 2:
            return 0

        mean = sum(lengths) / len(lengths)
        var = sum((x - mean) ** 2 for x in lengths) / len(lengths)  # population std
        return math.sqrt(var)


    def num_employers(self, person):
        organisations = set()

        for job in person:
            org = self._norm(job.get("organization"))
            if org:  # ignore empty / None
                organisations.add(org)

        return len(organisations)


    def job_switch_rate(self, person):
        career_months = self.career_length(person)
        if career_months == 0:
            return 0

        years = career_months / 12
        return self.num_of_jobs(person) / years


    def avg_jobs_per_employer(self, person):
        n_emp = self.num_employers(person)

        if n_emp == 0: return 0
        return self.num_of_jobs(person) / n_emp


    def internal_promotion_ratio(self, person):
        history = []

        for job in person:
            if job.get("status") != "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                sen = self._seniority_value(job.get("seniority"))
                org = self._norm(job.get("organization"))
                if start and sen is not None and org:
                    history.append((start, sen, org))

        if len(history) < 2: return 0

        history.sort(key=lambda x: x[0])

        promotions = 0
        internal_promotions = 0

        for i in range(1, len(history)):
            prev, curr = history[i-1], history[i]
            if curr[1] > prev[1]:   # promotion
                promotions += 1
                if curr[2] == prev[2]:
                    internal_promotions += 1

        if promotions == 0: return 0

        return internal_promotions / promotions


    def new_job_same_org(self, person):
        jobs = []

        for job in person:
            start = self._parse_year_month(job.get("startDate"))
            org = self._norm(job.get("organization"))
            if start and org:
                jobs.append((start, org, job.get("status")))

        if len(jobs) < 2:
            return 0

        jobs.sort(key=lambda x: x[0])

        prev_jobs = [j for j in jobs if j[2] != "ACTIVE"]

        if not prev_jobs: return 0

        last_prev = prev_jobs[-1]
        current_jobs = [j for j in jobs if j[2] == "ACTIVE"]

        if not current_jobs: return 0

        current_main = max(current_jobs, key=lambda x: x[0])

        return int(last_prev[1] == current_main[1])


    def num_jobs_current_employer(self, person):
        ref_org = None
        # To accept jobs just started
        max_duration = -1

        for job in person:
            if job.get("status") == "ACTIVE":
                start = self._parse_year_month(job.get("startDate"))
                if start:
                    duration = self._months_between(start, self._parse_year_month("2026-01"))
                    if duration > max_duration:
                        max_duration = duration
                        ref_org = self._norm(job.get("organization"))

        if not ref_org: return 0
        return sum(1 for job in person if self._norm(job.get("organization")) == ref_org)


    def data_pipeline(self):
        """
        ...
        """
        rows = []

        for p_id, person in enumerate(self.annotated_data):

            # Handle retired people
            # Maybe use Seniority of last Active job -> As example -> 50+ Persons are lost
            try:
                label = self.current_seniority_id(person)

                row = {
                    #"ID": p_id, # DataFrame Objekte werden default nummeriert daher ist Id eigentlich nicht notwendig
                    "Seniority (Label 1)": label,
                    #"Department (Label 2)": "",
                    "num_of_jobs": self.num_of_jobs(person),
                    "career_length (months)": self.career_length(person),
                    #"is_active": self.is_active(person), # Always 1 as Retired people are skipped
                    "avg_job_duration": self.avg_job_duration(person),
                    "current_job_duration": self.current_job_duration(person),
                    "number_of_active_jobs": self.num_active_jobs(person),
                    "time_since_last_job_change": self.time_since_job_change(person),
                    "longest_job_duration": self.longest_job_duration(person),
                    "std_job_duration": self.std_job_duration(person),
                    "num_employers": self.num_employers(person),
                    "job_switch_rate (years)": self.job_switch_rate(person),
                    "avg_jobs_per_employer": self.avg_jobs_per_employer(person),
                    #"internal_promotion_ratio": self.internal_promotion_ratio(person), # Potential data leakage
                    "new_job_same_org": self.new_job_same_org(person),
                    "num_jobs_current_employer": self.num_jobs_current_employer(person)
                }
                rows.append(row)
            except ValueError:
                continue

            # Normalisieren oder Standardisieren nicht vergessen

        self.data = pd.DataFrame(rows)
        self.Y = self.data[self.label_col]
        self.X = self.data.drop(columns=[self.label_col])
