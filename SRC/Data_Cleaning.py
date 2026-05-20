import pandas as pd
import numpy as np

# =============================================================================
# LOAD DATASETS
# =============================================================================
# Update these paths to point to your local CSV files
LOYALTY_PATH  = "/content/Customer Loyalty History.csv"
ACTIVITY_PATH = "/content/Customer Flight Activity.csv"

history_df  = pd.read_csv(LOYALTY_PATH)
activity_df = pd.read_csv(ACTIVITY_PATH)

print("── Raw shapes ──────────────────────────────────────")
print(f"  Loyalty History : {history_df.shape}")
print(f"  Flight Activity : {activity_df.shape}")

# =============================================================================
# SECTION 1 – CUSTOMER LOYALTY HISTORY
# =============================================================================

# 1A. Salary imputation
# ---------------------------------------------------------------
# Fill missing salaries with the median for each Education tier.
# Using median rather than mean makes the imputation robust to the
# high-earner outliers that are common in loyalty programme data.
history_df["Salary"] = (
    history_df
    .groupby("Education")["Salary"]
    .transform(lambda x: x.fillna(x.median()))
)

# Safety net: if an entire Education bucket has no salary data at all,
# fall back to the global median so we never leave NaNs behind.
global_salary_median = history_df["Salary"].median()
history_df["Salary"] = history_df["Salary"].fillna(global_salary_median)

# 1B. Cancellation date columns
# ---------------------------------------------------------------
# Active members have NaN here; cancelled members store the date as
# a float (e.g. 2018.0 / 3.0).  Replace NaNs with 0, then cast to
# int so downstream comparisons work cleanly (0 == "not cancelled").
for col in ["Cancellation Year", "Cancellation Month"]:
    if col in history_df.columns:
        history_df[col] = history_df[col].fillna(0).astype(int)

# 1C. Derive a membership status flag
# ---------------------------------------------------------------
# A Boolean column is much cheaper to filter on than checking two
# integer columns every time.
history_df["Is_Active"] = (history_df["Cancellation Year"] == 0)

# 1D. Standardise string columns
# ---------------------------------------------------------------
# Strip accidental whitespace and normalise case on every object column
# so joins and group-bys don't split on invisible differences.
str_cols = history_df.select_dtypes(include="object").columns
history_df[str_cols] = (
    history_df[str_cols]
    .apply(lambda s: s.str.strip().str.title())
)

print("\n── Loyalty History – null counts after cleaning ────")
print(history_df.isnull().sum()[history_df.isnull().sum() > 0]
      .rename("remaining nulls")
      .to_string() or "  None – all columns fully populated ✓")

# =============================================================================
# SECTION 2 – CUSTOMER FLIGHT ACTIVITY
# =============================================================================

# 2A. Exact duplicate rows
# ---------------------------------------------------------------
# Inspection of the raw file reveals 1 922 fully identical rows
# (same Loyalty Number, Year, Month AND all metrics identical).
# These are data-pipeline artefacts and should be dropped outright.
before = len(activity_df)
activity_df = activity_df.drop_duplicates()
print(f"\n── Flight Activity – exact duplicates removed: {before - len(activity_df)} rows")

# 2B. Non-identical same-period duplicates
# ---------------------------------------------------------------
# After removing exact duplicates, 1 949 additional rows share the
# same (Loyalty Number, Year, Month) key but carry different metric
# values – these look like split records from a booking amendment.
# Strategy: aggregate them by summing the additive metrics so we end
# up with exactly one row per customer-month.
ADDITIVE_COLS = [
    "Total Flights",
    "Distance",
    "Points Accumulated",
    "Points Redeemed",
    "Dollar Cost Points Redeemed",
]
before = len(activity_df)
activity_df = (
    activity_df
    .groupby(["Loyalty Number", "Year", "Month"], as_index=False)[ADDITIVE_COLS]
    .sum()
)
print(f"  Split-record duplicates collapsed: {before - len(activity_df)} rows")
print(f"  Rows after deduplication: {len(activity_df)}")

# 2C. Guard against negative values
# ---------------------------------------------------------------
# The raw file currently has no negatives, but clipping is kept as a
# defensive step – refund or system-error entries can appear in future
# loads and should never produce negative flight counts or distances.
for col in ADDITIVE_COLS:
    activity_df[col] = activity_df[col].clip(lower=0)

# 2D. Fix Points Accumulated dtype
# ---------------------------------------------------------------
# The column loads as float64 (likely due to a NaN in an earlier data
# version).  Cast to int now that negatives are clipped and no NaNs
# remain so arithmetic downstream behaves consistently.
activity_df["Points Accumulated"] = activity_df["Points Accumulated"].astype(int)

# 2E. Build a proper datetime index
# ---------------------------------------------------------------
# Combining Year + Month into a Pandas Period (calendar month) is more
# semantically correct than a day-anchored Timestamp when the data has
# no actual day component.  Use Period for period-arithmetic (e.g.
# "how many months since enrolment") and keep a Timestamp version for
# tools that don't support Period (e.g. most plotting libraries).
activity_df["Period"] = pd.to_datetime(
    activity_df["Year"].astype(str) + "-" + activity_df["Month"].astype(str)
).dt.to_period("M")
activity_df["Date"] = activity_df["Period"].dt.to_timestamp()

# 2F. Derived efficiency metric
# ---------------------------------------------------------------
# Points per km flown – useful for segmentation; avoids division by
# zero by replacing 0-distance rows with NaN.
activity_df["Points_Per_KM"] = np.where(
    activity_df["Distance"] > 0,
    activity_df["Points Accumulated"] / activity_df["Distance"],
    np.nan,
)

print("\n── Flight Activity – null counts after cleaning ────")
print(activity_df.isnull().sum()[activity_df.isnull().sum() > 0]
      .rename("remaining nulls")
      .to_string() or "  None (Points_Per_KM NaN only where Distance = 0) ✓")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n── Final shapes ────────────────────────────────────")
print(f"  Loyalty History : {history_df.shape}")
print(f"  Flight Activity : {activity_df.shape}")
print("\nData is clean and ready for analysis.")