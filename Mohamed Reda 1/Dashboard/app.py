import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hotel Booking Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\CRIZMA\Desktop\Cellelua\first_task\Dashboard\data.csv")

df = load_data()

st.title("🏨 Hotel Booking Dashboard")
st.markdown("A flexible dashboard (Plotly-free!) to explore your hotel booking dataset 📊")

# Sidebar
st.sidebar.header("🔍 Filters")

market_segments = df['market segment type'].dropna().unique().tolist()
booking_statuses = df['booking status'].dropna().unique().tolist()

segment_filter = st.sidebar.multiselect("Market Segment Type", market_segments, default=market_segments)
status_filter = st.sidebar.multiselect("Booking Status", booking_statuses, default=booking_statuses)
repeated_filter = st.sidebar.selectbox("Repeated Customers", options=["All", "Yes", "No"])
lead_time_range = st.sidebar.slider("Lead Time", int(df['lead time'].min()), int(df['lead time'].max()), (0, int(df['lead time'].max())))

filtered_df = df[
    df['market segment type'].isin(segment_filter) &
    df['booking status'].isin(status_filter) &
    df['lead time'].between(lead_time_range[0], lead_time_range[1])
]

if repeated_filter == "Yes":
    filtered_df = filtered_df[filtered_df['repeated'] == 1]
elif repeated_filter == "No":
    filtered_df = filtered_df[filtered_df['repeated'] == 0]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Bookings", len(filtered_df))
col2.metric("Confirmed", (filtered_df['booking status'] == 'Confirmed').sum())
col3.metric("Canceled", (filtered_df['booking status'] == 'Canceled').sum())

st.subheader("📊 Booking Status by Market Segment")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.countplot(data=filtered_df, x="market segment type", hue="booking status", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

if 'P-C' in filtered_df.columns and 'repeated' in filtered_df.columns:
    st.subheader("🔥 Heatmap: Market Segment × Repeated → Avg(P-C)")
    heatmap_data = filtered_df.groupby(['market segment type', 'repeated'])['P-C'].mean().unstack()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="vlag", ax=ax2)
    st.pyplot(fig2)

st.subheader("⏳ Lead Time Distribution")
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.histplot(data=filtered_df, x="lead time", bins=30, kde=True, ax=ax3)
st.pyplot(fig3)

st.subheader("📈 Correlation Heatmap")
numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_cols) > 1:
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    corr = filtered_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4, linewidths=0.5)
    st.pyplot(fig4)
else:
    st.warning("Not enough numeric data for correlation heatmap.")

with st.expander("🔍 Show Filtered Data"):
    st.dataframe(filtered_df)

st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name='filtered_bookings.csv',
    mime='text/csv',
)
