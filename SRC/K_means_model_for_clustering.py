import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 1. Load the scored dataset (must contain the Churn_Probability from previous step)
df = pd.read_csv("master_churn_scored.csv")

# 2. Engineer the Custom Framework Features
# Handle division by zero using np.where
df['CLV_Per_Month'] = np.where(df['Tenure_Months'] > 0, df['CLV'] / df['Tenure_Months'], 0)

# Assuming a 24-month observation window as stated in the framework
df['Frequency_Penetration'] = df['Months_Flown'] / 24.0

# Engagement Intensity: Flights per month, but ONLY for the months they actually flew
df['Engagement_Intensity'] = np.where(df['Months_Flown'] > 0, df['Total_Flights_Ever'] / df['Months_Flown'], 0)

# Rename Months_Inactive to match the framework's terminology
df['Recency_Months'] = df['Months_Inactive']
df['Recency_Months'] = df['Recency_Months'].clip(upper=24) # Cap at 24 for stability as requested

# 3. Define the 10 dimensions for clustering
segment_features = [
    "CLV",
    "CLV_Per_Month",
    "Tenure_Months",
    "Frequency_Penetration",
    "Engagement_Intensity",
    "Redemption_Rate",
    "Recency_Months",
    "Salary",
    "Card_Tier_Ordinal",
    "Churn_Probability"
]

# Ensure clean data for the algorithm
clustering_data = df.dropna(subset=segment_features).copy()
X_cluster = clustering_data[segment_features]

# 4. Standardize the Features (Critical for K-Means distance calculations)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 5. Train the K-Means Model (k=5 based on the framework)
# n_init=10 runs the algorithm 10 times with different centroids and picks the best one
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clustering_data['Cluster_ID'] = kmeans.fit_predict(X_scaled)

# 6. Map the numeric clusters to the Persona Names
# Note: K-Means assigns IDs (0-4) randomly based on the random_state.
# You will need to check the averages and update this dictionary to match the correct Persona to the correct ID.
persona_mapping = {
    0: "VIP Loyalists",
    1: "Emerging Members",
    2: "Loyal Budget Flyers",
    3: "At-Risk Dormant",
    4: "High-Engagement Newcomers"
}

clustering_data['Customer_Persona'] = clustering_data['Cluster_ID'].map(persona_mapping)

# Merge back to the main dataframe
df = df.merge(clustering_data[['Loyalty Number', 'Cluster_ID', 'Customer_Persona']], on='Loyalty Number', how='left')

# 7. Output the Cluster Profiles for Verification
# This allows you to verify that Cluster 0 actually matches the VIP profile, etc.
profile_summary = df.groupby('Customer_Persona')[segment_features].mean().round(2)
profile_summary['Count'] = df['Customer_Persona'].value_counts()
profile_summary['% of Base'] = (df['Customer_Persona'].value_counts(normalize=True) * 100).round(1)

print("\n--- FINAL CLUSTER PROFILES ---")
print(profile_summary.T.to_string())

# Save the final, fully segmented dataset
df.to_csv("master_final_segmented.csv", index=False)