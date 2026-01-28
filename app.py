import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===============================
# APP TITLE & DESCRIPTION
# ===============================
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

st.markdown(
    "👉 *Discover hidden customer groups without predefined labels.*"
)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("Wholesale customers data.csv")

numerical_features = [
    'Fresh',
    'Milk',
    'Grocery',
    'Frozen',
    'Detergents_Paper',
    'Delicassen'
]

# ===============================
# SIDEBAR – INPUT SECTION
# ===============================
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1",
    numerical_features
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    numerical_features,
    index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=4
)

random_state = st.sidebar.number_input(
    "Random State (optional)",
    min_value=0,
    value=42,
    step=1
)

run_button = st.sidebar.button("🟦 Run Clustering")

# ===============================
# VALIDATION
# ===============================
if feature_1 == feature_2:
    st.warning("⚠️ Please select **two different features**.")
    st.stop()

# ===============================
# RUN CLUSTERING
# ===============================
if run_button:
    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # ===============================
    # VISUALIZATION SECTION
    # ===============================
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        df[feature_1],
        df[feature_2],
        c=df['Cluster'],
        cmap='viridis',
        s=60
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        s=250,
        marker='X',
        label='Cluster Centers'
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Clusters")
    ax.legend()

    st.pyplot(fig)

    # ===============================
    # CLUSTER SUMMARY SECTION
    # ===============================
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby('Cluster')[[feature_1, feature_2]]
        .agg(['count', 'mean'])
    )

    summary.columns = ['_'.join(col) for col in summary.columns]
    st.dataframe(summary)

    # ===============================
    # BUSINESS INTERPRETATION
    # ===============================
    st.subheader("💡 Business Interpretation")

    for cluster_id in sorted(df['Cluster'].unique()):
        avg_1 = df[df['Cluster'] == cluster_id][feature_1].mean()
        avg_2 = df[df['Cluster'] == cluster_id][feature_2].mean()

        if avg_1 > df[feature_1].mean() and avg_2 > df[feature_2].mean():
            insight = "High-spending customers across multiple categories"
            emoji = "🟢"
        elif avg_1 < df[feature_1].mean() and avg_2 < df[feature_2].mean():
            insight = "Budget-conscious customers with low annual spend"
            emoji = "🟡"
        else:
            insight = "Moderate spenders with selective purchasing behavior"
            emoji = "🔵"

        st.write(f"{emoji} **Cluster {cluster_id}:** {insight}")

    # ===============================
    # USER GUIDANCE BOX
    # ===============================
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.info("⬅️ Select features and click **Run Clustering** to begin.")
