from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# =============================================================================
# 0. FEATURE ENGINEERING (Calculate missing columns)
# =============================================================================
# Calculate lifetime metrics from activity_df
agg_activity = activity_df.groupby("Loyalty Number").agg(
    Total_Distance_Ever=("Distance", "sum"),
    Total_Points_Acc=("Points Accumulated", "sum"),
    Months_Flown=("Total Flights", lambda x: (x > 0).sum()),
    Avg_Flights_Per_Month=("Total Flights", "mean")
).reset_index()

# Calculate Tenure and Card Tier Ordinal
master_df["Enrollment_Date"] = pd.to_datetime(master_df["Enrollment Year"].astype(str) + "-" + master_df["Enrollment Month"].astype(str) + "-01")
master_df["Tenure_Months"] = ((CURRENT_DATE - master_df["Enrollment_Date"]).dt.days / 30.44).fillna(0).astype(int)

tier_map = {"Star": 1, "Nova": 2, "Aurora": 3}
master_df["Card_Tier_Ordinal"] = master_df["Loyalty Card"].map(tier_map)

# Fix: Clean up existing columns before re-merging to prevent KeyError/Suffixes
num_cols_to_fill = ["Total_Distance_Ever", "Total_Points_Acc", "Months_Flown", "Avg_Flights_Per_Month"]
master_df = master_df.drop(columns=[c for c in num_cols_to_fill if c in master_df.columns])

# Merge and fillna ONLY on numeric columns
master_df = master_df.merge(agg_activity, on="Loyalty Number", how="left")
master_df[num_cols_to_fill] = master_df[num_cols_to_fill].fillna(0)

master_df["Redemption_Rate"] = (master_df["Total_Redeemed_Ever"] / master_df["Total_Points_Acc"].replace(0, np.nan)).fillna(0)

# =============================================================================
# 1. DEFINE LEAKAGE-FREE FEATURES
# =============================================================================
NUMERIC_FEATURES = [
    "Salary", "CLV", "Tenure_Months", "Total_Flights_Ever",
    "Total_Redeemed_Ever", "Total_Distance_Ever", "Total_Points_Acc",
    "Months_Flown", "Avg_Flights_Per_Month", "Redemption_Rate", "Card_Tier_Ordinal"
]

CATEGORICAL_FEATURES = [
    "Province", "Gender", "Education", "Marital Status", "Enrollment Type"
]

X = master_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = master_df["Churn_Flag"]

# =============================================================================
# 2. PREPROCESSING PIPELINE
# =============================================================================
categorical_pipe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
numeric_pipe = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# =============================================================================
# 3. TRAIN / TEST SPLIT & INITIALIZE BALANCED RANDOM FOREST
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=20,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
)

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])
model.fit(X_train, y_train)

# =============================================================================
# 4. GENERATE CHURN PROBABILITIES & RISK BUCKETS
# =============================================================================
master_df["Churn_Probability"] = model.predict_proba(X)[:, 1].round(4)
master_df["Risk_Segment"] = pd.cut(
    master_df["Churn_Probability"],
    bins=[0, 0.30, 0.60, 1.01],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    right=False
)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print(f"ROC-AUC Score: {roc_auc:.3f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 7. Extract Feature Importances
importances = model.named_steps['classifier'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Drivers of Churn ---")
print(importance_df.head(5).to_string(index=False))

# Save the entire master_df, not just output_cols, so all features are available for downstream tasks.
master_df.to_csv("master_churn_scored.csv", index=False)

print("\n── Risk segment distribution ───────────────────────────────")
print(master_df.groupby("Risk_Segment", observed=True)["Churn_Flag"].agg(Count="count", Actual_Churners="sum").assign(Churn_Rate_pct=lambda d: (d["Actual_Churners"] / d["Count"] * 100).round(1)))
print("\n✓ Saved → master_churn_scored.csv")