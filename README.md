# K-Means Clustering — Step-by-Step Visualization

> Course project for **Data Mining** (7th semester) — Department of Informatics and Telecommunications, University of Ioannina.

## Problem Description

This project implements a full step-by-step version of the **K-Means clustering algorithm**, visualizing the clustering process after every update. The algorithm is applied on synthetic data from three multivariate Gaussian distributions. 

The implementation follows the format `[ClusterCenters, IDC] = mykmeans(Data, K)` as required. Intermediate states (centroids and labels) are stored in a list and used to generate dynamic visualizations of the clustering progress.

## Key Features

- **Custom K-Means implementation**
  - Random centroid initialization without replacement
  - Convergence check based on centroid movement (`epsilon = 1e-3`)
  - Euclidean distance-based label assignment
  - Mean-based centroid update
- **SSE Computation**
  - Sum of Squared Errors calculated at each update step
- **Visualization**
  - Cluster plots after every iteration
  - Final result plot
  - SSE plot per iteration

## How to Run

```bash
git clone https://github.com/konrantos/kmeans-clustering-visualization.git
cd kmeans-clustering-visualization
python kmeans_visual.py
```

> Requires Python 3 and the libraries `numpy` and `matplotlib`.

## Output Example

The script will display:
- **Update plots** of clusters and centroids at each iteration
- A **line chart** showing SSE per update step
- **Console output** of centroids and SSE values at each step

```
Cluster centers per step:
Step 1:
[[...], [...], [...]]
...
Step N:
[[...], [...], [...]]

SSE per step:
Step 1: 183.52
...
Step N: 43.92
```

## Dataset

- **Synthetic Gaussian data**
- 3 clusters × 50 points = 150 points total
- Random seed fixed with `np.random.seed(0)` for reproducibility

## License

MIT License

## Acknowledgements

- University of Ioannina — course project for *Data Mining*
