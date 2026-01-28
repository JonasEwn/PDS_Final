HF-profile: https://huggingface.co/Luu200

# Predicting Domain and Seniority from LinkedIn Data

This repository contains the code and data for the **Practical Data Science Final Project**.
The goal in this assignmnet is to predict the **professional domain (department)** and **seniority level** of individuals based on LinkedIn CV data using machine learning approaches.

The repository is organised according to the structure of the task description, with our different approaches categorised accordingly.

---

## Repository Structure

```
PDS_Final/
├── data/
│   ├── dep_PL.csv
│   ├── department-v2.csv
│   ├── highest_active_jobs.csv
│   ├── labeled_not_annotated.csv
│   ├── linkedin-cvs-annotated.json
│   ├── linkedin-cvs-not-annotated.json
│   ├── sen_PL.csv
│   └── seniority-v2.csv
│
├── ensemble/
│   ├── department.ipynb
│   ├── seniority.ipynb
│   └── plots/
│
├── task_1_rule_based/
│   └── rule-based.ipynb
│
├── task_2_embeddings/
│   └── embedding.ipynb
│
├── task_3_finetuning/
│   └── setfit/
│       ├── dep_sf.ipynb
│       └── sen_sf.ipynb
│
├── task_4_programmatic_labeling/
│   ├── Department/
│   │   └── dep_gem_api.ipynb
│   └── Seniority/
│       └── gem_api.ipynb
│
├── task_6/
│   ├── plots/
│   ├── preprocessing/
│   ├── bow_department.ipynb
│   ├── bow_seniority.ipynb
│   ├── tf_idf_department.ipynb
│   └── tf_idf_seniority.ipynb
│
└── README.md
```

## File Overview

### data/

Contains all datasets used in the project, both the provided and those created by us.

- **dep_PL.csv:** Dataset was created from the annotated.json. Contains newest Position of each Person and the associated Department
- **department-v2.csv:** Dataset provided
- **highest_active_jobs.csv:** Dataset was created from the not-annotated.json. Contains newest Position of each Person and the associated Department and Seniority
- **labeled_not_annotated.csv:**
- **linkedin-cvs-annotated.json:** Dataset provided
- **linkedin-cvs-not-annotated.json:** Dataset provided
- **sen_PL.csv:** Dataset was created from the annotated.json. Contains newest Position of each Person and the associated Seniority
- **seniority-v2.csv:** Dataset provided
