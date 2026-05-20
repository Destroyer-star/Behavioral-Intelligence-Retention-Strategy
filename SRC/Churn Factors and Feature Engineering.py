# =============================================================================
# CONFIG (Assuming history_df and activity_df are already clean and loaded)
# =============================================================================
CURRENT_DATE = pd.Timestamp("2018-12-01")
INACTIVITY_THRESHOLD_MONTHS = 12

# =============================================================================
# 1. BUILD PER-CUSTOMER BEHAVIOURAL FEATURES
# =============================================================================
def last_active_date(group: pd.DataFrame, col: str) -> pd.Timestamp:
    """Return the most recent Date on which `col` > 0, or NaT."""
    rows = group.loc[group[col] > 0, "Date"]
    return rows.max() if not rows.empty else pd.NaT

customer_features = (
    activity_df # type: ignore
    .groupby("Loyalty Number")
    .apply(
        lambda g: pd.Series({
            "Last_Flight_Date":     last_active_date(g, "Total Flights"),
            "Last_Redemption_Date": last_active_date(g, "Points Redeemed"),
            "Total_Flights_Ever":   g["Total Flights"].sum(),
            "Total_Redeemed_Ever":  g["Points Redeemed"].sum(),
        }),
        include_groups=False
    )
    .reset_index()
)

# Last Active Date = whichever of the two is more recent
customer_features["Last_Active_Date"] = (
    customer_features[["Last_Flight_Date", "Last_Redemption_Date"]]
    .max(axis=1)
)

def months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)

customer_features["Months_Inactive"] = customer_features["Last_Active_Date"].apply(
    lambda d: months_between(CURRENT_DATE, d) if pd.notna(d) else 999
)

# =============================================================================
# 2. MERGE INTO MASTER DATASET
# =============================================================================
master_df = pd.merge(history_df, customer_features, on="Loyalty Number", how="left") # type: ignore

# Fix: Create 'Cancellation_Date' column after merging
# This column is needed for churn rules but was not created in history_df.
master_df["Cancellation_Date"] = pd.NaT # Initialize with NaT
cancelled_mask = master_df["Cancellation Year"] != 0
master_df.loc[cancelled_mask, "Cancellation_Date"] = pd.to_datetime(
    master_df.loc[cancelled_mask, "Cancellation Year"].astype(str) + '-' +
    master_df.loc[cancelled_mask, "Cancellation Month"].astype(str) + '-01'
)

# =============================================================================
# 3. DERIVED FLAGS NEEDED BY THE CHURN RULES
# =============================================================================
master_df["Flew_After_Cancel"] = (
    master_df["Last_Flight_Date"].notna()
    & master_df["Cancellation_Date"].notna()
    & (master_df["Last_Flight_Date"] > master_df["Cancellation_Date"])
)

master_df["Is_Cancelled"] = master_df["Cancellation_Date"].notna()

# =============================================================================
# 4. MULTI-FACTOR CHURN LOGIC
# =============================================================================
master_df["Churn_Flag"]   = 0
master_df["Churn_Reason"] = "Active – recent activity"

# ── Rule 1 (Silent Churn)
silent_churn_mask = master_df["Months_Inactive"] >= INACTIVITY_THRESHOLD_MONTHS
master_df.loc[silent_churn_mask, "Churn_Flag"]   = 1
master_df.loc[silent_churn_mask, "Churn_Reason"] = f"Silent churn – inactive >={INACTIVITY_THRESHOLD_MONTHS} months"

# ── Rule 2 (Hard Exit)
hard_exit_mask = master_df["Is_Cancelled"] & ~master_df["Flew_After_Cancel"]
master_df.loc[hard_exit_mask, "Churn_Flag"]   = 1
master_df.loc[hard_exit_mask, "Churn_Reason"] = "Hard exit – cancelled, no post-cancel flights"

# ── Rule 3 (Exception - Overrides 1 & 2)
exception_mask = master_df["Is_Cancelled"] & master_df["Flew_After_Cancel"]
master_df.loc[exception_mask, "Churn_Flag"]   = 0
master_df.loc[exception_mask, "Churn_Reason"] = "Exception – cancelled but still flying"

# =============================================================================
# 5. DROP INTERMEDIATE COLUMNS
# =============================================================================
master_df.drop(
    columns=["Last_Flight_Date", "Last_Redemption_Date", "Last_Active_Date", "Flew_After_Cancel", "Is_Cancelled"],
    inplace=True,
)

# Print Summary
print(master_df["Churn_Flag"].value_counts(normalize=True).mul(100).round(2).rename(index={0: "Active (0)", 1: "Churned (1)"}))