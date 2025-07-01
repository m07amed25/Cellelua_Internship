import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Setup page
st.set_page_config(page_title="Hotel Booking Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\CRIZMA\Desktop\Cellelua\First Task\first inten project.csv")
    return df

df = load_data()
st.title("🏨 Hotel Booking Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Dynamic filters based on data columns
market_segments = df['market segment type'].dropna().unique().tolist()
booking_statuses = df['booking status'].dropna().unique().tolist()

segment_filter = st.sidebar.multiselect("Market Segment Type", market_segments, default=market_segments)
status_filter = st.sidebar.multiselect("Booking Status", booking_statuses, default=booking_statuses)
repeated_filter = st.sidebar.selectbox("Repeated Customers", options=["All", "Yes", "No"])
lead_time_range = st.sidebar.slider("Lead Time", int(df['lead time'].min()), int(df['lead time'].max()), (0, int(df['lead time'].max())))

# Apply filters
filtered_df = df[
    df['market segment type'].isin(segment_filter) &
    df['booking status'].isin(status_filter) &
    (df['lead time'].between(lead_time_range[0], lead_time_range[1]))
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

# Booking status count per market segment
st.subheader("📊 Booking Status by Market Segment")
fig1 = px.histogram(filtered_df, x='market segment type', color='booking status', barmode='group')
st.plotly_chart(fig1, use_container_width=True)

# Heatmap: Market segment × repeated → avg(P-C)
if 'P-C' in filtered_df.columns and 'repeated' in filtered_df.columns:
    st.subheader("🔥 Heatmap: Market Segment × Repeated → Avg(P-C)")
    heatmap_data = filtered_df.groupby(['market segment type', 'repeated'])['P-C'].mean().unstack()
    fig2, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="vlag", ax=ax)
    st.pyplot(fig2)

# Lead time distribution
st.subheader("⏳ Lead Time Distribution")
fig3 = px.histogram(filtered_df, x='lead time', nbins=30)
st.plotly_chart(fig3, use_container_width=True)

# Correlation heatmap (numerical features only)
st.subheader("📈 Correlation Heatmap")
numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
corr = filtered_df[numeric_cols].corr()

fig4, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig4)

# Raw data display
with st.expander("🔍 Show Filtered Data"):
    st.dataframe(filtered_df)

# Download filtered data
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name='filtered_bookings.csv',
    mime='text/csv',
)

