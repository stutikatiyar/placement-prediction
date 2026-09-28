# ⚡ Student Placement Prediction System
### *Enterprise-Grade Student Placement Readiness & Compensation Band Engine*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-00f2fe?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

---

## 📌 Executive Summary

**NEXUS CareerPulse AI** is an end-to-end predictive analytics and decision support platform designed to evaluate candidate placement readiness and project compensation tiers (4.0 LPA to 20.0 LPA) during institutional campus recruitment drives.

Standard classification algorithms tend to output extreme binary step probabilities (0.0 or 1.0) and overlook mandatory, non-negotiable enterprise placement criteria. NEXUS bridges this gap by integrating:
1. **Calibrated Ensemble Inference**: 150-tree Random Forest classifier regularized with leaf smoothing (`min_samples_leaf=15`) to output smooth, continuous probability estimates.
2. **Corporate Eligibility Logic**: Strict Zero-Active-Backlog enforcement and 60.0% secondary board aggregate thresholds (10th and 12th Grade).
3. **5-Pillar Modular Rubric**: Granular evaluation spanning Language ($L_x$), Aptitude ($A_x$), Core Engineering ($C_x$), Programming ($P_x$), and Soft Skills ($S_x$).
4. **Diagnostic Visualizations & Interpretability**: Integrated Confusion Matrix heatmaps and Gini Feature Importance rankings directly inside the dashboard.
5. **Interactive High-Performance UI**: Dark-mode cybernetic dashboard featuring inline vector rendering (zero external font dependencies), candidate auto-population by USN/Name, and glassmorphic navigation routing.

---

## 🏗️ System Architecture

```text
                                 [ Candidate Record Input ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ USN Directory Lookup ]                     [ Manual Profile Entry ]
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               [ Corporate Policy Filters ]
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             Active Backlogs > 0?                       10th or 12th Board < 60%?
              ├── YES ──► [ DISQUALIFIED (0.0%) ]        ├── YES ──► [ SEVERE RISK CRITICAL (2-14%) ]
              └── NO                                     └── NO
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                                 [ Scikit-Learn Pipeline ]
                                  ├── One-Hot Encoding
                                  ├── Standard Scaler
                                  └── Calibrated Random Forest
                                             │
                                             ▼
                                 [ Continuous Probability ]
                                 [   & CGPA Elasticity    ]
                                             │
                                             ▼
                                [ Tier Band Recommendation ]
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
                  Mass Recruiter       Mid-Tier Product       Tier-1 MNC
                  (4.0 - 5.0 LPA)      (8.0 - 14.0 LPA)    (14.0 - 20.0 LPA)
```

---

## 📊 5-Pillar Departmental Assessment Rubric

Candidates are evaluated across 5 modular clearance categories. To qualify for corporate placement drives, candidates must clear a minimum Level 3 threshold across all domains:

| Module Code | Skill Domain | Score Levels | Corporate Cutoff Benchmark |
| --- | --- | --- | --- |
| **$L_x$** | Language & Professional Fluency | $L_0 - L_4$ | Minimum $L_3$ clearance |
| **$A_x$** | Quantitative & Logical Aptitude | $L_0 - L_4$ | Minimum $L_3$ clearance |
| **$C_x$** | Core Engineering Fundamentals | $L_0, L_2 - L_5$ | Minimum $L_3$ clearance |
| **$P_x$** | Algorithmic Coding & Development | $L_{0.0} - L_{5.0}$ | Minimum $L_{3.0}$ clearance |
| **$S_x$** | Soft Skills & Executive Presence | $L_0 - L_4$ | Minimum $L_3$ clearance |

---

## 📈 Model Performance & Comparative Benchmark

The models were evaluated against multiple supervised baselines using a stratified 80:20 train-test split ($N = 2,400$ test records):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Calibrated Ensemble)** | **76.33%** | **77.48%** | **82.05%** | **0.7970** | **Production Deployed** |
| **Gradient Boosting Classifier (GBM)** | **76.17%** | **77.38%** | **81.82%** | **0.7954** | Evaluated Baseline |
| **Logistic Regression** | **74.88%** | **76.00%** | **81.31%** | **0.7856** | Linear Baseline |
| **K-Nearest Neighbors ($k=9$)** | **71.42%** | **72.63%** | **79.47%** | **0.7590** | Distance-based Baseline |

### Confusion Matrix (Random Forest Test Set, $N = 2,400$)

```text
                          Predicted: Not Placed (0)    Predicted: Placed (1)
Actual: Not Placed (0)              717 (TN)                    324 (FP)
Actual: Placed (1)                  244 (FN)                   1,115 (TP)
```

> **Data Integrity & Leakage Prevention**: Non-predictive identifiers (`student_id`, `student_name`, `usn`) and composite score targets (`Overall_Level_Score`, `package_lpa`, `company_type`) were strictly isolated and removed prior to model training to prevent target leakage.

---

## 🔍 Key Placement Determinants (Feature Importance)

The Random Forest ensemble model quantifies the relative importance of attributes determining placement success:

| Rank | Feature | Importance | Interpretation & Business Impact |
| :---: | :--- | :---: | :--- |
| **1** | **CGPA** | **42.92%** | Primary academic gateway for corporate eligibility screening. |
| **2** | **Coding Skills** | **12.67%** | Determines capability for higher-tier technical roles. |
| **3** | **Active Backlogs** | **10.91%** | Decisive disqualification penalty across campus drives. |
| **4** | **Aptitude Score** | **8.38%** | Governs round-1 online assessment clearing rates. |
| **5** | **$P_x$ Level Reached** | **3.77%** | Coding & development rubric tier clearance. |
| **6** | **Internships** | **3.63%** | Practical workplace and applied technical experience. |
| **7** | **Projects** | **3.04%** | Technical capstone depth and portfolio strength. |
| **8** | **Communication Skills** | **1.96%** | Influences HR and technical interview evaluations. |
| **9** | **Certifications** | **1.95%** | Specialized continuous domain learning credentials. |
| **10**| **$S_x$ Level Reached** | **1.74%** | Soft skills and executive presence rubric level. |

> **Fairness & Non-Bias**: Demographic variables (`gender`, `degree`, `branch`) contribute $< 0.7\%$ each, ensuring evaluations are meritocratic and objective.

---

## ⚙️ Business Rules & Diagnostic Engine

1. **Mandatory Zero-Backlog Policy**:
   * If `backlogs > 0`, the candidate is immediately marked `DISQUALIFIED` with `0.0%` probability, regardless of test tier clearances or CGPA standing.

2. **60.0% Secondary Board Filter**:
   * If either 10th or 12th board marks fall below `60.0%`, placement probability is constrained to `2.0% - 14.0%` due to automated enterprise applicant tracking system (ATS) filters.

3. **CGPA Elasticity**:
   * High CGPA ($\ge 8.5$) unlocks Tier-1 product roles (14.0 – 20.0 LPA).
   * Mid CGPA (7.0 – 7.4) scales placement probability downward to mirror real-world market competition.
   * CGPA below 6.0 triggers first-class cutoff exclusion alerts.

4. **Prescriptive Remediation**:
   * For non-placed students, the diagnostic engine isolates the deficient assessment domain ($L_x, A_x, C_x, P_x, S_x$) and prescribes targeted skill-building areas.

---

## 📂 Project Repository Structure

```text
placement-prediction/
├── data/
│   └── student_placement_data.csv        # Integrated candidate records & metrics (12,000 rows)
├── models/
│   ├── best_pipeline.pkl                 # Serialized Scikit-Learn pipeline
│   ├── benchmark_results.json            # Dynamic model comparison evaluation metrics
│   ├── confusion_matrices.json           # Raw confusion matrices for all algorithms
│   ├── feature_importances.json          # Gini feature importance rankings
│   ├── confusion_matrix.png              # Confusion matrix heatmap visualization
│   └── feature_importance.png            # Top 10 feature importances bar chart
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb         # Feature analysis & distribution curves
│   └── 02_model_training.ipynb           # Model selection, tuning & serialization
├── app.py                                # Streamlit dashboard application with dynamic benchmarks
├── train.py                              # End-to-end model training, evaluation & asset exporter
├── generate_data.py                      # Synthetic candidate generator based on 5-pillar rubric
├── update_data.py                        # Metadata populator for candidate names and USN codes
├── requirements.txt                      # Production runtime dependencies
├── .gitignore                            # Virtual environment & cache filters
└── README.md                             # Technical project documentation
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites

* Python 3.10 or 3.11
* Git CLI

### 1. Clone the Repository

```bash
git clone https://github.com/Sujalranjan/placement-prediction.git
cd placement-prediction
```

### 2. Set Up Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Runtime Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train Models & Export Assets (Optional)

```bash
python train.py
```

### 5. Run the Web Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 👩‍💻 Authors & Contact

* **Developers**: Sujal Ranjan & Stuti Katiyar
* **Repository**: [https://github.com/Sujalranjan/placement-prediction](https://github.com/Sujalranjan/placement-prediction)