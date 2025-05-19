import numpy as np
import matplotlib.pyplot as plt

# List to store intermediate states of the algorithm (ClusterCenters and labels) for each iteration
iterations_states = []


# Data: N×M matrix (N samples, M features), where each row corresponds to a sample
# K: number of clusters to identify
def mykmeans(Data, K):
    epsilon = 1e-3  # Convergence threshold

    N = len(Data)  # Number of samples
    M = len(Data[0])  # Number of features

    # Randomly select K unique samples as initial centroids
    initial_indices = np.random.choice(N, K, replace=False)
    ClusterCenters = Data[initial_indices, :]

    IDC = np.zeros(N, dtype=int)  # Array to store cluster label for each sample

    while True:
        distances = np.zeros((N, K))
        for i in range(K):
            diff = Data - ClusterCenters[i]
            distances[:, i] = np.sum(diff**2, axis=1)

        new_IDC = np.argmin(distances, axis=1)

        if np.array_equal(new_IDC, IDC):
            break

        IDC = new_IDC

        new_centers = np.zeros((K, M))
        for i in range(K):
            cluster_points = Data[IDC == i]
            if len(cluster_points) > 0:
                new_centers[i] = np.mean(cluster_points, axis=0)
            else:
                new_centers[i] = ClusterCenters[i]

        shift = np.sum((ClusterCenters - new_centers) ** 2)
        ClusterCenters = new_centers
        iterations_states.append((ClusterCenters.copy(), IDC.copy()))

        if shift < epsilon:
            break

    return ClusterCenters, IDC


# Set seed for reproducibility
np.random.seed(0)
N_per_cluster = 50

# Mean and covariance for each Gaussian distribution
mean1 = [4.0, 0.0]
cov1 = [[0.29, 0.4], [0.4, 4.0]]

mean2 = [5.0, 7.0]
cov2 = [[0.29, 0.4], [0.4, 0.9]]

mean3 = [7.0, 4.0]
cov3 = [[0.64, 0.0], [0.0, 0.64]]

# Generate samples
X1 = np.random.multivariate_normal(mean1, cov1, N_per_cluster)
X2 = np.random.multivariate_normal(mean2, cov2, N_per_cluster)
X3 = np.random.multivariate_normal(mean3, cov3, N_per_cluster)

# Combine all into one dataset
Data = np.vstack((X1, X2, X3))

# Run K-means for K=3
K = 3
final_centers, labels = mykmeans(Data, K)

# Compute SSE at each step
sse_per_iteration = []
all_states = iterations_states + [(final_centers, labels)]

for centers, lbls in all_states:
    SSE = 0
    for i in range(K):
        cluster_points = Data[lbls == i]
        diff = cluster_points - centers[i]
        SSE += np.sum(diff**2)
    sse_per_iteration.append(SSE)

# Visualize clustering progress
colors = ["red", "green", "blue"]
step = 1
for centers, lbls in all_states:
    plt.figure(figsize=(6, 4))
    for j in range(K):
        cluster_points = Data[lbls == j]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[j],
            alpha=0.7,
            label=f"Cluster {j+1}",
        )
    plt.scatter(
        centers[:, 0], centers[:, 1], color="black", marker="+", s=300, label="Centers"
    )
    if step < len(all_states):
        plt.title(f"K-means Update Step {step}")
    else:
        plt.title("Final K-means Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    step += 1

# SSE plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(sse_per_iteration) + 1), sse_per_iteration, marker="o")
for i, sse in enumerate(sse_per_iteration, start=1):
    plt.text(i, sse, f"{sse:.2f}", fontsize=10, ha="center", va="bottom")
plt.title("SSE per Update Step")
plt.xlabel("Step")
plt.ylabel("SSE")
plt.grid(True)
plt.show()

# Print cluster centers at each step
print("\nCluster centers per step:")
for step, (centers, _) in enumerate(all_states, start=1):
    print(f"Step {step}:")
    print(centers)

# Print SSE per step
print("\nSSE per step:")
for step, sse in enumerate(sse_per_iteration, start=1):
    print(f"Step {step}: {sse}")
