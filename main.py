from ml_t6.preprocessing import Preprocessing

pre = Preprocessing(
    X_link="/Users/jonas/Documents/Master_Vorlesungen/Semester_02/Practical Data Science/Final/PDS_Final/data/linkedin-cvs-annotated.json",
    seniority_link  = "/Users/jonas/Documents/Master_Vorlesungen/Semester_02/Practical Data Science/Final/PDS_Final/data/seniority-v2.csv",
    department_link = "/Users/jonas/Documents/Master_Vorlesungen/Semester_02/Practical Data Science/Final/PDS_Final/data/department-v2.csv"
)

pre.data_pipeline()
print(f"Data Frame:\n{pre.data}")

from xgboost import XGBClassifier
