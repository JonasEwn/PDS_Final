HF-profile: https://huggingface.co/Luu200

# Predicting Domain and Seniority from LinkedIn Data

This repository contains the code and data for the **Practical Data Science Final Project**.
The goal in this assignmnet is to predict the **professional domain (department)** and **seniority level** of individuals based on LinkedIn CV data using machine learning approaches.

The repository is organised according to the structure of the task description, with our different approaches categorised accordingly.

**Note: In order for the notebooks to run some file paths may need to be adjusted.**

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

- **dep_PL.csv:** Dataset was created from not-annotated.json. Contains newest Position of each Person and the associated Department.
- **department-v2.csv:** Provided dataset.
- **highest_active_jobs.csv:** Dataset was created from annotated.json. Contains newest Position of each Person and the associated Department and Seniority.
- **labeled_not_annotated.csv:** Dataset was created from not-annotated.json. Contains newest Position of each Person and the associated Department and Seniority.
- **linkedin-cvs-annotated.json:** Dataset provided.
- **linkedin-cvs-not-annotated.json:** Provided dataset.
- **sen_PL.csv:** Dataset was created from not-annotated.json. Contains newest Position of each Person and the associated Seniority.
- **seniority-v2.csv:** Provided dataset.

### ensemble/

The ensemble folder holds the ensemble models made from the Models created in Task 3 & 6.

- **department.ipynb:** Complete Pipeline for Department Soft-Voting Model.
- **seniority.ipynb:** Complete Pipeline for Seniority Soft-Voting Model.
- **plots/ :** Folder holds plots used in our report.

### task_1_rule_based/

Contains the Rule-Based Baseline model.

- **rule-based.ipynb:** Code for Seniority and Department prediction with a rule-based model.

### task_2_embeddings/

Folder for Embedding-based labeling.

- **embedding.ipynb:** Code for Seniority and Department scoring with a embedding.

### task_3_finetuning/SetFit/

Fine Tuned Model from Huggingface. The trained models are available [here](https://huggingface.co/Luu200).

- **dep_sf.ipynb:** Complete Pipeline for Department classification.
- **sen_sf.ipynb:** Complete Pipeline for Seniority classification.

### task_4_programmatic_labeling/

Holds Code for Task 4.

- **Department/dep_gem_api.ipynb:** Code for Department prediction via programmatic labeling.
- **Seniority/gem_api.ipynb:** Code for Seniority prediction via programmatic labeling.

### task_6/

Folder contains the Code for (Bag-of-Word and TF-IDF) + Logistic Regression Classification. Uses both architectures to predict Seniority and Department.

- **plots/ :** Folder holds plots used in our report.
- **preprocessing/ :** Contains now unused files two preprocess the raw .csv and .json file. Use classes and methods are directly inside each Notebook
- **bow_department.ipynb:** Complete Pipeline for BoW + LR training and classification of Department. Also explain different features and classes using SHAP
- **bow_seniority.ipynb:** Complete Pipeline for BoW + LR training and classification of Seniority. Also explain different features and classes using SHAP
- **/tf_idf_department.ipynb:** Complete Pipeline for TF-IDF + LR training and classification of Department.
- **/tf_idf_seniority.ipynb:** Complete Pipeline for TF-IDF + LR training and classification of Seniority.
