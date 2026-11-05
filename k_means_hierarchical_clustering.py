# K-Means Clustering on Sales Data
# (The original repository file name mentions hierarchical clustering, but the
# supplied program implements K-Means only.)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Dataset
DATA_PATH = Path(__file__).resolve().parent / "dataset" / "sales_data_sample.csv"
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

print("First 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# 2. Select Numerical Features
# We use sales-related numerical attributes
features = [
    "QUANTITYORDERED",
    "PRICEEACH",
    "SALES"
]

X = df[features]

# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())

# 3. Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 5. Apply K-Means
# Select K based on the elbow point
k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# 6. Display Cluster Information
print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Centers:")
centers = scaler.inverse_transform(kmeans.cluster_centers_)

cluster_centers = pd.DataFrame(
    centers,
    columns=features
)

print(cluster_centers)

# 7. Display Sample Clustered Data
print("\nClustered Data:")
print(df[features + ["Cluster"]].head(20))

# 8. Visualize Clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    df["QUANTITYORDERED"],
    df["SALES"],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.title("K-Means Clustering of Sales Data")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# EXPLANATION (added):
# 1. The script groups sales records into clusters using three numerical
#    columns: quantity ordered, price per item, and total sales.
# 2. StandardScaler converts these columns to comparable scales before
#    distance-based clustering.
# 3. The Elbow Method fits K-Means for K values 1 through 10 and records
#    inertia (the total within-cluster squared distance). The plot helps select
#    a sensible number of clusters.
# 4. The script uses K = 3, assigns every row a Cluster label, and prints the
#    cluster counts and centers. Centers are converted back to original units.
# 5. The final scatter plot displays quantity against sales, colored by cluster.
# Complexity: roughly O(iterations * rows * K * selected_features) per fit.
#
# RUN ON LINUX:
#   cd sem7_annotated
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
#   python3 ml/k_means_hierarchical_clustering.py
# Dataset required: ml/dataset/sales_data_sample.csv (download_datasets.sh).
# Note: this source performs K-Means; it does not implement hierarchical
# clustering despite the original repository file name.
# -----------------------------------------------------------------------------
