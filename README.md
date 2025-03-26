# RNA-Clash-Correction
 
# Principal Submanifold for clustering and RNA correction

This is a Python implementation of our sampling method described in our paper *A Principal Submanifold-Based Approach for Clustering and Multiscale RNA Correction*.

## Pre-require
- Python 3.8
- scikit-learn                 1.3.2
- scipy                        1.10.1
- numpy                        1.24.4
- matplotlib                   3.7.5
- colorcet (optional, for additional colormaps)
- Jupyter Notebook (or JupyterLab) for running the simulation notebook
You can install the required packages using pip:

```bash
pip install numpy scipy scikit-learn matplotlib colorcet jupyter
```

## PSM-Based Dimensionality Reduction and Clustering

This repository demonstrates a framework for dimensionality reduction and clustering using a Point Set Manifold (PSM) algorithm. The code is organized into three main files:

- **PSM.py**  
  Contains the core function `psm(sample, h, maxiter, rho, e_n, alpha=1)`, which implements the PSM algorithm for iteratively updating sample points based on their neighborhood.

- **utils.py**  
  Provides a set of helper functions including:
  - Angle conversion functions (`angle_to_cos_sin` and `cos_sin_to_angle`)
  - Distance calculations for toroidal (angle) data
  - Data generators for simulation experiments (for both dimensionality reduction and clustering)
  - Evaluation metrics and plotting utilities (e.g., `plot_k_distance`)

- **simulation.ipynb**  
  A Jupyter Notebook that contains two simulation sections:
  1. **Dimensionality Reduction Simulation:**  
     - Generates synthetic data (with adjustable features, noise level, and case type) using functions from `utils.py`.
     - Applies the PSM algorithm from `PSM.py` (after converting angles to cosine-sine pairs) to reduce the dimensionality.
     - Uses t-SNE to visualize the reduced data in 2D.
  2. **Clustering Simulation:**  
     - Generates high-dimensional clustering data via `utils.data_generator_cl`.
     - Applies dimensionality reduction with the PSM algorithm.
     - Performs clustering using DBSCAN and reclassifies noise points with k-Nearest Neighbors.
     - Visualizes the clustering results (using t-SNE for high-dimensional data) and evaluates clustering performance using metrics such as Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

## Customization

- **PSM Parameters:**  
  You can adjust parameters in `PSM.py` such as the neighborhood radius (`h`), maximum iterations (`maxiter`), convergence threshold (`rho`), and the number of singular vectors (`e_n`). These control how the PSM algorithm updates the sample points.

- **Data Generation:**  
  Modify the parameters (e.g., `n_features`, `case`, `n_samples`, `noise_level`) in `utils.py` or directly within the notebook to experiment with different synthetic datasets.

- **Clustering Settings:**  
  The DBSCAN parameters (`eps` and `min_samples`) and the kNN settings can be tuned within the clustering section of the notebook to adjust clustering performance.
  

