import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# 1. PAGE CONFIGURATION & THEME
# =============================================================================
# Set a wide layout for a more professional dashboard feel
st.set_page_config(page_title="Airline Loyalty Command Center", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# 2. DATA LOADING (Cached for performance)
# =============================================================================
@st.cache_data
def load_data():
    # Load the fully scored and segmented dataset
    df = pd.read_csv("master_final_segmented.csv")
    return df

df = load_data()

# Ensure we have active users for targeting (Exclude the At-Risk Dormant/Churned if needed, but let's keep all for overview)
# Define the order for our risk segments so they sort nicely
risk_order = ["Low Risk", "Medium Risk", "High Risk"]
df['Risk_Segment'] = pd.Categorical(df['Risk_Segment'], categories=risk_order, ordered=True)

# =============================================================================
# 3. SIDEBAR: THE CONTROL PANEL
# =============================================================================
st.sidebar.title("Targeting Controls")
st.sidebar.markdown("Filter the customer base to deploy retention strategies.")

# Filter by Persona
selected_persona = st.sidebar.selectbox(
    "Select Customer Segment (Persona):",
    ["All Segments"] + list(df['Customer_Persona'].dropna().unique())
)

# Filter by Risk Level
selected_risk = st.sidebar.multiselect(
    "Select Churn Risk Level:",
    options=risk_order,
    default=["High Risk", "Medium Risk"]
)

# Optional: Filter by specific Loyalty Tier
selected_tier = st.sidebar.multiselect(
    "Select Loyalty Tier:",
    options=df['Loyalty Card'].unique(),
    default=list(df['Loyalty Card'].unique())
)

# Apply Filters
filtered_df = df.copy()
if selected_persona != "All Segments":
    filtered_df = filtered_df[filtered_df['Customer_Persona'] == selected_persona]
if selected_risk:
    filtered_df = filtered_df[filtered_df['Risk_Segment'].isin(selected_risk)]
if selected_tier:
    filtered_df = filtered_df[filtered_df['Loyalty Card'].isin(selected_tier)]

# =============================================================================
# 4. MAIN DASHBOARD: EXECUTIVE SUMMARY (KPIs)
# =============================================================================
st.title("Loyalty Retention & Value Command Center")
st.markdown("Use this interface to identify at-risk cohorts and trigger strategic interventions.")
st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Target Audience Size", f"{len(filtered_df):,}")
with col2:
    avg_clv = filtered_df['CLV'].mean() if not filtered_df.empty else 0
    st.metric("Avg. Cohort CLV", f"${avg_clv:,.0f}")
with col3:
    # Revenue at Risk (Sum of CLV for the selected cohort)
    rev_at_risk = filtered_df['CLV'].sum() if not filtered_df.empty else 0
    st.metric("Total Revenue at Risk", f"${rev_at_risk:,.0f}")
with col4:
    avg_risk = filtered_df['Churn_Probability'].mean() * 100 if not filtered_df.empty else 0
    st.metric("Avg. Churn Probability", f"{avg_risk:.1f}%")

st.divider()

# =============================================================================
# 6. STRATEGY EXECUTION: THE PLAYBOOK & ACTIONABLE DATA
# =============================================================================
st.divider()
st.subheader("Retention Playbook & Actionable Target List")

# 1. Define the exact PLG Strategies we engineered
strategy_map = {
    "VIP Loyalists": "Deploy 'Zero-Friction Guarantee' & Status Portability passes.",
    "Emerging Members": "Drop 5k 'Phantom Points' (60-day expiry) to force first flight.",
    "Loyal Budget Flyers": "Unlock 'In-Flight Micro-Burns' to clear point liability.",
    "At-Risk Dormant": "Trigger Route-Specific Price Alerts or a 'Prove It' status match.",
    "High-Engagement Newcomers": "Activate 'Fast-Track Multiplier' (3x points for 90 days)."
}

# 2. Add the strategy directly to the dataset as a new column
if not filtered_df.empty:
    filtered_df['Recommended_Action'] = filtered_df['Customer_Persona'].map(strategy_map)

# 3. Display the prominent Playbook trigger based on the sidebar selection
if selected_persona != "All Segments":
    st.info(f"**PLAYBOOK TRIGGER FOR {selected_persona.upper()}:** {strategy_map.get(selected_persona, '')}")
else:
    st.markdown("*Showing all segments. Look at the **Recommended Action** column in the table below to see the specific strategy for each user.*")

# 4. Display the actionable raw data table
if not filtered_df.empty:
    # We include our new 'Recommended_Action' column here
    display_cols = ['Loyalty Number', 'Customer_Persona', 'Risk_Segment', 'Churn_Probability', 
                    'Recommended_Action', 'CLV', 'Avg_Flights_Per_Month']
    
    # Display the dataframe, sorted by highest churn risk first
    st.dataframe(
        filtered_df[display_cols].sort_values(by='Churn_Probability', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Mock Export Button
    st.button("Export Cohort & Trigger Campaigns in Salesforce 🚀")

# =============================================================================
# 6. STRATEGY EXECUTION: THE ACTIONABLE DATA
# =============================================================================
st.subheader("Actionable Target List")

# Provide dynamic strategic advice based on the selected persona
if selected_persona == "VIP Loyalists":
    st.info("**PLAYBOOK TRIGGER:** Deploy the 'Zero-Friction Guarantee' and 'Status Portability' pass to this cohort immediately.")
elif selected_persona == "Emerging Members":
    st.info("**PLAYBOOK TRIGGER:** Drop 'Phantom Points' (5k) with a 60-day expiry to force the first flight habit.")
elif selected_persona == "Loyal Budget Flyers":
    st.info("**PLAYBOOK TRIGGER:** Unlock 'In-Flight Micro-Burns' to clear point liability off the balance sheet.")
elif selected_persona == "At-Risk Dormant":
    st.error("**PLAYBOOK TRIGGER:** Trigger Route-Specific Price Alerts or a 90-day 'Prove It' status match challenge.")
elif selected_persona == "High-Engagement Newcomers":
    st.success("**PLAYBOOK TRIGGER:** Activate the 'Fast-Track Multiplier' (3x points) to lock them in before momentum fades.")
else:
    st.write("Select a specific Persona from the sidebar to view targeted retention strategies.")

# Display the raw data for export/action
if not filtered_df.empty:
    display_cols = ['Loyalty Number', 'Customer_Persona', 'Risk_Segment', 'Churn_Probability', 
                    'CLV', 'Avg_Flights_Per_Month', 'Redemption_Rate', 'Tenure_Months']
    
    # Sort by highest risk first
    st.dataframe(
        filtered_df[display_cols].sort_values(by='Churn_Probability', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Mock Export Button (Required by the prompt to "close the gap to business action")
    st.button("Export Cohort to Salesforce Marketing Cloud 🚀")