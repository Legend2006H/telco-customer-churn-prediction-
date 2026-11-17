# Telco Churn Intelligence Platform — CyberPulse AI

A complete, production-grade Telecom Churn Prediction & Executive Analytics platform built with Python, Scikit-Learn, Streamlit, and Plotly. Styled with a high-performance **Cyberpunk / Gen-Z Dark Aesthetic** featuring glassmorphism, glowing micro-animations, and an all-inclusive **Home Command Center**.

---

## Verified Dataset & Benchmark Numbers

| Metric / Check | Value / Output | Status |
|---|---|---|
| **Raw Dataset Dimensions** | 7,043 rows × 21 columns | Verified ✅ |
| **Blank TotalCharges Rows** | 11 rows (`" "`, all `tenure = 0`) dropped | Verified ✅ |
| **Cleaned Dataset Dimensions** | 7,032 rows × 20 columns | Verified ✅ |
| **Engineered Features** | 37 features (One-Hot Encoded with `drop_first=True`) | Verified ✅ |
| **Train / Test Split** | 5,625 train / 1,407 test (80/20 stratified, seed 42) | Verified ✅ |
| **Test Confusion Matrix Sum** | 920 (TN) + 113 (FP) + 178 (FN) + 196 (TP) = 1,407 | Verified ✅ |
| **Best Model** | Logistic Regression (ROC-AUC = 0.8347) | Verified ✅ |
| **Model Accuracy** | **0.7932 (79.32%)** (+5.8% over No-Churn baseline 73.46%) | Verified ✅ |
| **Model Precision** | **0.6343 (63.43%)** | Verified ✅ |
| **Model Recall (Sensitivity)** | **0.5241 (52.41%)** | Verified ✅ |
| **Model F1-Score** | **0.5739** | Verified ✅ |
| **Model Specificity** | **0.8906 (89.06%)** | Verified ✅ |

### Complete 6-Classifier Benchmark Leaderboard
| Model | Accuracy | ROC-AUC | Precision | Recall (Sensitivity) | F1-Score | Specificity | Brier Loss |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Best)** | **0.7932** | **0.8347** | **0.6343** | **0.5241** | **0.5739** | **0.8906** | **0.1361** |
| Random Forest | 0.7846 | 0.8158 | 0.6156 | 0.5053 | 0.5551 | 0.8858 | 0.1417 |
| Naive Bayes | 0.6745 | 0.8121 | 0.4403 | 0.8289 | 0.5751 | 0.6186 | 0.2079 |
| Support Vector Machine | 0.7882 | 0.7901 | 0.6407 | 0.4626 | 0.5373 | 0.9061 | 0.1468 |
| K-Nearest Neighbors | 0.7555 | 0.7755 | 0.5403 | 0.5374 | 0.5389 | 0.8345 | 0.1691 |
| Decision Tree | 0.7164 | 0.6412 | 0.4675 | 0.4813 | 0.4743 | 0.8015 | 0.2836 |

---

## Complete Graph & Visualization Catalog

### 1. ⚡ Command Center (Home)
- **8-Tile Mega Metric Deck**: Model Accuracy (`79.32%`), ROC-AUC (`0.8347`), Churn Rate (`26.5%`), Verified Accounts (`7,032`), Monthly MRR at Risk (`$139.1k/mo`), Recall (`52.41%`), Precision (`63.43%`), Avg Tenure (`32.4 mos`).
- **Graph 1 — Customer Retention Split**: Neon cyber donut chart with center percentage callout.
- **Graph 2 — Tenure Survival Decay Curve**: Dual histogram overlay showing attrition density over account lifetime.
- **Graph 3 — Monthly Recurring Revenue (MRR) Exposure Flow**: Comparing Total MRR, Retained MRR, Churned MRR, and Targeted 40% Recoverable MRR.
- **Graph 4 — Billing Density Curves**: Split violin distributions comparing Monthly Charges for retained vs churned accounts.
- **Graph 5 — Top 4 Churn Multipliers**: Horizontal bar chart identifying highest risk concentrations.
- **Table — 6-Model Leaderboard**: Quick-glance benchmark table directly on the home page.
- **Executive Action Banner**: 4 tactical retention recommendations.

### 2. 📊 Exploratory Data (EDA)
- **Graph 6 — Hierarchical Customer Journey Sunburst**: Interactive multi-level drill-down (`Contract → InternetService → PaymentMethod → Churn`).
- **Graph 7 — 3D Feature Interaction Space**: Interactive rotatable 3D scatter plot (`Tenure × MonthlyCharges × TotalCharges`).
- **Graph 8 — Tenure Density Distribution**: Binned histogram with marginal box plots.
- **Graph 9 — Monthly Charges Boxplot & Outliers**: Segmented by churn status.
- **Graph 10 — Total Charges vs. Tenure Scatter Matrix**: With OLS regression trendline.
- **Graph 11 — Bivariate Correlation Heatmap**: Matrix for all continuous & numerical features.

### 3. 🔥 Churn Drivers & Triggers
- **Graph 12 — Churn Rate by Contract Commitment**: Month-to-month vs 1-Year vs 2-Year.
- **Graph 13 — Churn Rate by Payment Channel**: Electronic check vs credit card/bank transfer.
- **Graph 14 — Churn Rate by Internet Technology**: Fiber optic vs DSL vs No internet.
- **Graph 15 — Churn Rate by Lifecycle Tenure Group**: 0-12m, 13-24m, 25-48m, 49-72m.
- **Graph 16 — Service Add-on Shield**: Grouped bars for all 6 add-on services showing churn with vs without.
- **Graph 17 — Service Adoption Radar Chart**: 8-axis spider radar comparing feature adoption between retained vs churned cohorts.

### 4. 🎯 Model Benchmark & Accuracy
- **Graph 18 — ROC Curves (All 6 Models)**: Combined receiver operating characteristic curves.
- **Graph 19 — Precision-Recall (PR) Curves**: All 6 models with baseline no-skill rate.
- **Graph 20 — Cumulative Gains / Targeting Lift Chart**: Demonstrates how targeting the top 30% of accounts captures over 68% of all churners.
- **Graph 21 — Annotated 2x2 Confusion Matrix**: With raw sample counts, percentages, and performance breakdowns.
- **Table — Full Model Benchmark Suite**: Including Brier loss, Specificity, and F1.

### 5. ⚡ Feature Signals & Weights
- **Graph 22 — Random Forest Gini Importance**: Top 10 non-linear features.
- **Graph 23 — Logistic Regression Absolute Weights**: Top 10 linear coefficients.
- **Graph 24 — Odds Ratio Multipliers Chart (e^β)**: Quantifies exactly how many times each feature multiplies or divides the odds of churn.

### 6. 🔮 Customer Risk Predictor & What-If Lab
- **Graph 25 — Real-Time Radial Risk Gauge**: 0% to 100% dial with color-coded risk bands.
- **Graph 26 — Counterfactual "What-If" Simulation**: Compares current customer risk against alternative retention scenarios (e.g. 1-year contract, free tech support, autopay bundle).
- **Automated Retention Playbook**: Generates custom offers based on profile inputs.

### 7. 🎚️ Threshold Simulator & ROI Engine
- **Graph 27 — Threshold Sensitivity Curves**: Simultaneous plot of Accuracy, Recall, Precision, and F1 across thresholds 0.05 to 0.95.
- **Graph 28 — Real-Time Confusion Matrix**: Recalculates dynamically with the slider.
- **💰 Campaign Net Profit Calculator**: Live economic calculator determining net financial savings.

### 8. 👥 Live Customer Explorer
- Filter by contract, internet type, payment channel, and churn status.
- Sortable cohort view with 1-click **CSV Export**.

---

## How to Run

```bash
cd "c:\Users\hrush\OneDrive\Documents\Prediction model dashbaord\Telco-Customer-Churn"
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.
