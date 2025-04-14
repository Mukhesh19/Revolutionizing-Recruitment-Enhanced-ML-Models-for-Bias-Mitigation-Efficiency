# Revolutionizing Recruitment: Bias-Mitigated Machine Learning for Fair & Efficient Hiring

## 📘 Project Overview

This project proposes an ethically grounded, machine learning-driven framework for transforming recruitment decisions. It uses an enriched dataset of **1,500 candidate profiles** to evaluate and improve hiring outcomes through **predictive analytics**, **bias mitigation**, and **explainable AI (XAI)**.

Key innovations include:
- Integrating fairness-aware models (e.g., **Exponentiated Gradient**)
- Applying **SHAP** and **LIME** for global and local interpretability
- Using ensemble models like **CatBoost**, **XGBoost**, and **Random Forest**
- Balancing performance with fairness metrics such as **Demographic Parity** and **Equalized Odds**

## 🔍 Research Objective

To optimize recruitment models that:
- Achieve high predictive accuracy
- Maintain fairness across gender
- Enhance interpretability for HR stakeholders

## 🧠 Models Used

- **Baseline**: Logistic Regression
- **Tree-based Models**: Random Forest, Extra Trees
- **Boosting Models**: CatBoost, XGBoost, LightGBM, AdaBoost
- **Deep Learning**: DNN (Multi-layer Perceptron with ReLU & Dropout)
- **Fairness-aware**: Exponentiated Gradient (via `Fairlearn`)

## 🧪 Methodology

### Data Preprocessing
- Categorical encoding for `Gender`, `EducationLevel`, `RecruitmentStrategy`
- Outlier inspection (boxplot) and preservation for realism
- Z-score scaling for distance-based models
- 30-fold Stratified Cross-Validation
- GridSearchCV & RandomSearchCV for tuning

### Feature Overview
- Features include Age, Gender, Education Level, Experience, Skill Score, Personality Score, Interview Score, Recruitment Strategy
- Target: `HiringDecision` (0 = Not Hired, 1 = Hired)

## 📈 Model Performance

| Model             | Accuracy | ROC-AUC |
|------------------|----------|---------|
| CatBoost          | 95.4%    | 0.93    |
| Random Forest     | 94.4%    | 0.93    |
| XGBoost           | 94.2%    | 0.92    |
| DNN, SVM, Extra Trees | 92% | 0.92    |
| Logistic Regression | 87%   | 0.92    |

> 🔎 **CatBoost** outperformed all models due to its efficient handling of categorical data and robust generalization.

## ⚖️ Fairness Metrics

| Model               | Demographic Parity (DPD) | Equalized Odds (EOD) |
|--------------------|--------------------------|----------------------|
| Logistic Regression | 0.0142                   | 0.0414               |
| Exponentiated Gradient | **~0.01**             | **~0.03**            |

> ✅ Bias mitigation through fairness-aware learning reduced disparities **without sacrificing accuracy**.

## 🧠 Explainability

### SHAP (Global)
- Most impactful features:
  - Recruitment Strategy (1.88)
  - Education Level (0.80)
  - Skill Score (0.80)
- Low impact: Gender (0.10), Age (0.09)

### LIME (Local)
- Most influential per-instance: Skill Score (0.1575)
- Recruitment Strategy ranked low in LIME, indicating structural impact over individual predictions.

## 📌 Key Insights

- **Recruitment Strategy** is the strongest predictor, indicating that company policy heavily affects hiring decisions.
- High **model agreement** across top classifiers (>92% accuracy) shows robustness.
- **Fairness-aware algorithms** are viable in production, preserving performance while reducing bias.
- **Explainable AI tools** are essential for trust, legal compliance, and model auditability.

## 📉 Limitations

- Dataset size (n = 1,500) limits generalizability
- Subjective factors like "cultural fit" not captured
- Interpretability for DNNs remains limited

## 🚀 Future Directions

- Incorporate **unstructured data** (e.g., resume text, psychometric results)
- Explore **transformers** and multimodal ML
- Evaluate **intersectional fairness**
- Deploy in live systems with **feedback loops** and audit trails

## 📚 References

This project builds upon literature by Barocas et al. (2019), Raji et al. (2020), Tasheva & Karpovich (2024), and Bellamy et al. (2019). For a complete list, refer to the full [report](./Full%20Report.pdf).

## 💡 Authors

- **Mukhesh Ravi**  
- **Harika Yenuga**  
- **Dr. Itauma Itauma**  
The Richard DeVos Graduate School of Management, Northwood University, USA

## 🛠️ Tools & Libraries

- Python (Scikit-learn, XGBoost, CatBoost, SHAP, LIME, Fairlearn)
- Matplotlib & Seaborn (Visualization)
- Jupyter Notebook
