import os
import sys
import math
import collections
import numpy as np
from numpy import array
import numpy.linalg as la
import numpy.random as arandom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import average, fcluster, single, ward, centroid, weighted, dendrogram, linkage
from scipy.optimize import linear_sum_assignment, brentq
from scipy.integrate import quad
from scipy.sparse.linalg import svds
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.metrics import confusion_matrix, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture
import colorcet as cc

PI = np.pi

def find_nbr(x, samples, h):
    """
    Find neighbors of x within radius h in samples.
    
    Parameters:
        x (numpy.ndarray): A point or set of points.
        samples (numpy.ndarray): Sample points (each column represents a point).
        h (float): The radius threshold.
    
    Returns:
        numpy.ndarray: The subset of samples that are within the given radius h.
    """
    distances = cdist(samples.T, np.atleast_2d(x.T))
    ind_nbr = np.any(distances < h + np.finfo(float).eps, axis=1)
    return samples[:, ind_nbr]

def angle_to_cos_sin(angle_data):
    """
    Convert angle values to their corresponding cosine and sine components.
    
    Parameters:
        angle_data (numpy.ndarray): Array of angles.
        
    Returns:
        numpy.ndarray: Array with cosine and sine pairs for each angle.
    """
    cos_sin_list = []
    for i in range(angle_data.shape[0]):
        cos_sin = []
        for j in range(angle_data.shape[1]):
            cos_sin = cos_sin + [np.cos(angle_data[i][j]), np.sin(angle_data[i][j])]
        cos_sin_list.append(cos_sin)
    cos_sin_list = np.array(cos_sin_list)
    return cos_sin_list

def cos_sin_to_angle(cos_sin):
    """
    Convert cosine and sine pairs back to angle values.
    
    Parameters:
        cos_sin (numpy.ndarray): Array with cosine and sine pairs.
        
    Returns:
        numpy.ndarray: Array of angles.
    """
    angle_list = []
    for i in range(cos_sin.shape[0]):
        angle = []
        for j in range(0, cos_sin.shape[1], 2):
            if cos_sin[i][j+1] >= 0:
                angle.append(np.arccos(cos_sin[i][j]))
            else:
                angle.append((2*PI - np.arccos(cos_sin[i][j])))
        angle_list.append(angle)
    angle_list = np.array(angle_list)
    return angle_list

def radian_to_degree(radian):
    """
    Convert radians to degrees.
    
    Parameters:
        radian (float): Angle in radians.
        
    Returns:
        float: Angle in degrees.
    """
    return radian * (180.0 / math.pi)

# Define the integrand function for arc length calculation
def integrand(x):
    return np.sqrt(1 + (4 * np.cos(2 * x))**2)

# Compute the total arc length over the interval [a, b]
def total_arclength(a, b):
    return quad(integrand, a, b)[0]

# Given an arc length value s, compute the corresponding x value in the interval [a, b]
def arclength_to_x(s, a, b):
    equation = lambda x: quad(integrand, a, x)[0] - s
    return brentq(equation, a, b)

def map_to_torus(samples, R=50, r=20):
    """
    Map the 2D samples in the plane to a 3D Torus.
    """
    theta, phi = samples[:, 0], samples[:, 1]  # Using samples as angular coordinates
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.vstack((x, y, z)).T


def distance_matrix(dihedral_angles):
    """
    Compute a distance matrix based on dihedral angle differences.
    
    Parameters:
        dihedral_angles (numpy.ndarray): Array of dihedral angles.
        
    Returns:
        numpy.ndarray: Distance matrix computed using the sum of squared minimal differences.
    """
    sum_dihedral_differences = np.zeros(int(dihedral_angles.shape[0] * (dihedral_angles.shape[0] - 1) / 2))
    for i in range(dihedral_angles.shape[1]):
        diff_one_dim = pdist(dihedral_angles[:, i].reshape((dihedral_angles.shape[0], 1)))
        # Using minimum difference in degrees (wrap-around at 360°)
        diff_one_dim = np.min((360 - diff_one_dim, diff_one_dim), axis=0) ** 2
        sum_dihedral_differences = sum_dihedral_differences + diff_one_dim
    return np.sqrt(sum_dihedral_differences)

def torus_distance(x1, x2):
    """
    Compute the angular distance between two sets of data points on a torus 
    for each dimension (without squaring or summing).
    
    Parameters:
        x1, x2 (numpy.ndarray): Arrays of shape (N, d).
        
    Returns:
        numpy.ndarray: Angular distances for each dimension.
    """
    diff = np.abs(x1 - x2)
    return np.minimum(diff, 2 * np.pi - diff)

def compute_min_distances(dataset_a, dataset_b):
    """
    Compute the minimum angular distance from each sample in dataset_a to any sample in dataset_b.
    The angular distance is defined as:
    
      d_{T}(x, y) = sqrt( sum_{j=1}^d ( min(|x_j - y_j|, 2π - |x_j - y_j|) )^2 )
    
    Parameters:
        dataset_a (numpy.ndarray): Array of shape (N_a, d) containing samples.
        dataset_b (numpy.ndarray): Array of shape (N_b, d) containing reference samples.
        
    Returns:
        min_distances (numpy.ndarray): Array of shape (N_a,) with the minimum distance for each sample in dataset_a.
        min_indices (numpy.ndarray): Array of shape (N_a,) with the indices of the closest sample in dataset_b.
    """
    # Compute angular distances for each pair using broadcasting
    diff = torus_distance(dataset_a[:, None, :], dataset_b[None, :, :])
    distances = np.sqrt(np.sum(diff**2, axis=2))
    # For each sample in dataset_a, find the closest sample in dataset_b
    min_indices = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(dataset_a.shape[0]), min_indices]
    return min_distances, min_indices

def plot_k_distance(data, min_samples=50):
    """
    Plot the k-distance graph for the dataset.
    
    Parameters:
        data (numpy.ndarray): Data points.
        min_samples (int): The minimum number of samples (k) to consider.
    """
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(data)

    distances, indices = neigh.kneighbors(data)
    k_dist = distances[:, min_samples-1]
    k_dist_sorted = np.sort(k_dist)

    plt.plot(k_dist_sorted)
    plt.ylabel('k-distance')
    plt.xlabel('Points sorted by distance')
    plt.title(f'K-distance Plot for min_samples = {min_samples}')
    plt.grid(True)
    plt.show()
    
def plot_torus_samples(samples, color='red'):
    """
    Plot a torus surface with overlaid sample points in a single color.
    
    Parameters:
        torus_samples (numpy.ndarray): Array of sample points mapped onto the torus,
                                       with shape (num_samples, 3).
        color (str or tuple): Color for the sample points.

    The function creates a 3D plot with a torus surface and scatters all the sample points 
    in the specified color.
    """
    torus_samples = map_to_torus(samples)
    # Create figure with constrained layout
    fig = plt.figure(figsize=(8, 12), constrained_layout=True)
    
    # Define torus parameters
    R = 50  # Major radius
    r = 20  # Minor (tube) radius
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    # Compute torus parameterization
    x = (R + r * np.cos(V)) * np.cos(U)
    y = (R + r * np.cos(V)) * np.sin(U)
    z = r * np.sin(V)
    
    # Create a 3D subplot and plot the torus surface
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='Greys', alpha=0.1, zorder=1)
    
    # Scatter all sample points on the torus using the specified color
    ax.scatter(
        torus_samples[:, 0],
        torus_samples[:, 1],
        torus_samples[:, 2],
        color=color,
        edgecolor='k',
        s=30,
        zorder=2
    )
    
    # Set the viewing angle
    ax.view_init(elev=30, azim=150)
    
    # Set axis limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    
    # Turn off the axes for a cleaner look
    ax.axis('off')
    plt.show()
    
    

def compute_approximation_error(subm, samples_without_noise):
    """
    Compute the approximation error based on the submatrix and noise-free samples.
    
    Parameters:
        subm (numpy.ndarray): Submatrix data.
        samples_without_noise (numpy.ndarray): Noise-free sample data.
    
    Returns:
        float: Mean approximation error.
    """
    distances_without_noise, _ = compute_min_distances(subm, samples_without_noise)
    approximation_error = np.mean(distances_without_noise)
    return approximation_error    

def compute_information_retention(n_features, samples, samples_fitted):
    """
    Compute the amount of information retained by each feature and the proportion of the total information.
    
    Parameters:
        n_features (int): Number of features.
        samples (numpy.ndarray): Original sample data.
        samples_fitted (list of numpy.ndarray): List of submatrix data arrays corresponding to each feature.
        
    Returns:
        list: Proportion of total information retained by each feature.
    """
    information_contained_psm_list = []
    
    for i in range(n_features):
        if i == 0:
            # Compare samples with the second-to-last array in samples_fitted
            distances = torus_distance(samples, samples_fitted[n_features - 2])
            info_value = np.sum(np.sum(distances**2, axis=1))
        elif i == n_features - 1:
            # Compare the first fitted sample with its mean
            mean_val = np.mean(samples_fitted[0], axis=0, keepdims=True)
            distances = torus_distance(samples_fitted[0], mean_val)
            info_value = np.sum(np.sum(distances**2, axis=1))
        else:
            # Compare consecutive fitted samples
            distances = torus_distance(samples_fitted[-i], samples_fitted[-(i + 1)])
            info_value = np.sum(np.sum(distances**2, axis=1))
            
        information_contained_psm_list.append(info_value)
        
    total_sum_psm = sum(information_contained_psm_list)
    proportion_of_information_retained_psm = [x / total_sum_psm for x in information_contained_psm_list]
    
    return proportion_of_information_retained_psm

def convert_clusters_to_arrays(clusters):
    # Get the unique cluster labels from the input array
    unique_clusters = np.unique(clusters)
    
    # For each unique cluster label, find the indices of elements belonging to that cluster
    cluster_arrays = [np.where(clusters == cluster_idx)[0] for cluster_idx in unique_clusters]
    
    # Return the list of arrays, where each array contains indices of one cluster
    return cluster_arrays

def closest_distance_between_lines(A1, B1, A2, B2):
    """
    Compute the shortest distance between two line segments in n-dimensional space.
    
    Args:
        A1, B1 (numpy.ndarray): Endpoints of the first line segment.
        A2, B2 (numpy.ndarray): Endpoints of the second line segment.
    
    Returns:
        min_distance (float): The minimum distance between the two line segments.
        closest_point_1 (numpy.ndarray): Closest point on the first line segment.
        closest_point_2 (numpy.ndarray): Closest point on the second line segment.
    """
    # Direction vectors for the line segments
    u = B1 - A1
    v = B2 - A2
    w0 = A1 - A2

    # Coefficients for the quadratic system
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    if denom != 0:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    else:
        # Lines are parallel; default to s=t=0
        s, t = 0, 0

    # Clamp s and t to the [0, 1] range
    s = max(0, min(1, s))
    t = max(0, min(1, t))

    closest_point_1 = A1 + s * u
    closest_point_2 = A2 + t * v
    min_distance = np.linalg.norm(closest_point_1 - closest_point_2)

    return min_distance, closest_point_1, closest_point_2

def generate_random_point(n_features):
    """
    Generate a random point in n-dimensional space within the range [pi/6, 11*pi/6].
    
    Parameters:
        n_features (int): Number of dimensions.
        
    Returns:
        numpy.ndarray: Random point coordinates.
    """
    return np.array([np.random.uniform(np.pi/6, 11 * np.pi/6) for _ in range(n_features)])

def generate_segments_with_min_distance(n_features, num_segments=3, min_distance=1.5, max_distance=2.0):
    """
    Generate `num_segments` line segments in n-dimensional space such that 
    the minimum distance between any two segments is greater than `min_distance`
    and less than `max_distance`.
    
    Parameters:
        n_features (int): Number of dimensions.
        num_segments (int): Number of segments to generate.
        min_distance (float): Minimum allowed distance between segments.
        max_distance (float): Maximum allowed distance between segments.
        
    Returns:
        list: A list of tuples, each containing two numpy.ndarray representing a segment's endpoints.
    """
    segments = []
    while len(segments) < num_segments:
        A = generate_random_point(n_features)
        B = generate_random_point(n_features)
        valid = True
        for (existing_A, existing_B) in segments:
            distance, _, _ = closest_distance_between_lines(A, B, existing_A, existing_B)
            if distance <= min_distance or distance >= max_distance:
                valid = False
                break
        if valid:
            segments.append((A, B))
    return segments

def generate_samples_on_line(start, end, n_samples):
    """
    Generate uniformly distributed samples along a line in n-dimensional space.
    
    Parameters:
        start (list or numpy.ndarray): Starting point coordinates.
        end (list or numpy.ndarray): Ending point coordinates.
        n_samples (int): Number of samples to generate.
        
    Returns:
        numpy.ndarray: Array of shape (n_samples, d) with the sample coordinates.
    """
    start = np.array(start)
    end = np.array(end)
    t_values = np.linspace(0, 1, n_samples)
    samples = start + np.outer(t_values, (end - start))
    return samples

def data_generator_dr(n_features, case, n_samples, noise_level):
    """
    Generate samples for simulations in dimensionality reduction.
    
    Parameters:
        n_features (int): Number of features/dimensions.
        case (int): Case number indicating the type of data to generate.
        n_samples (int): Number of samples to generate.
        noise_level (float): Standard deviation of Gaussian noise.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        X (numpy.ndarray): Noisy data samples.
        samples_without_noise (numpy.ndarray): Noise-free data samples.
    """
    np.random.seed(42)
    
    samples_without_noise = None
    X = None

    # 2D data
    if n_features == 2 and case == 1:
        x_values = np.linspace(1, 2 * np.pi - 1, n_samples)
        y_values = x_values.copy()
        samples_without_noise = np.column_stack((x_values, y_values))
        
    elif n_features == 2 and case == 2:
        a, b = 0, 2 * np.pi
        S = total_arclength(a, b)
        uniform_arclengths = np.random.uniform(0, S, n_samples)
        x_values = np.array([arclength_to_x(s, a, b) for s in uniform_arclengths])
        y_values = 2 * np.sin(2 * x_values) + np.pi
        samples_without_noise = np.column_stack((x_values, y_values))
        
    # 3D data
    elif n_features == 3 and case == 1:
        a1 = [1, 1, 1]
        a2 = [2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1]
        line1_start = np.array(a1)
        line1_end = np.array(a2)
        t = np.linspace(0, 1, n_samples)
        samples_without_noise = line1_start + t[:, np.newaxis] * (line1_end - line1_start)
        
    elif n_features == 3 and case == 2:
        a, b = 1, 5
        S = total_arclength(a, b)
        uniform_arclengths = np.random.uniform(0, S, n_samples)
        x1 = np.array([arclength_to_x(s, a, b) for s in uniform_arclengths])
        y1 = np.random.uniform(2.5, 3, n_samples)
        z1 = np.sin(x1 * 2) + np.pi
        samples_without_noise = np.column_stack([x1, y1, z1])
        
    # 4D data
    elif n_features == 4 and case == 1:
        start = [1, 1, 1, 1]
        end = [2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1]
        samples_without_noise = generate_samples_on_line(start, end, n_samples)
        
    elif n_features == 4 and case == 2:
        x1 = np.random.uniform(1, 5, n_samples)
        x2 = np.random.uniform(2.5, 3, n_samples)
        x3 = np.sin(x1 * 2) + np.pi
        x4 = (x1 + x2 + x3) / 3
        samples_without_noise = np.column_stack([x1, x2, x3, x4])
        
    # 5D data
    elif n_features == 5 and case == 1:
        start = [1, 1, 1, 1, 1]
        end = [2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1]
        samples_without_noise = generate_samples_on_line(start, end, n_samples)
        
    elif n_features == 5 and case == 2:
        x1 = np.random.uniform(1, 5, n_samples)
        x2 = np.random.uniform(2.5, 3, n_samples)
        x3 = np.sin(x1 * 2) + np.pi
        x4 = np.cos(x2 * 2) + np.pi
        x5 = (x1 + x2 + x3 + x4) / 4
        samples_without_noise = np.column_stack([x1, x2, x3, x4, x5])
        
    # 6D data
    elif n_features == 6 and case == 1:
        start = [1, 1, 1, 1, 1, 1]
        end = [2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1]
        samples_without_noise = generate_samples_on_line(start, end, n_samples)
        
    elif n_features == 6 and case == 2:
        x1 = np.random.uniform(1, 5, n_samples)
        x2 = np.random.uniform(2.5, 3, n_samples)
        x3 = np.sin(x1 * 2) + np.pi
        x4 = np.cos(x2 * 2) + np.pi
        x5 = np.sin(x4 * 2) + np.pi
        x6 = (x1 + x2 + x3 + x4 + x5) / 5
        samples_without_noise = np.column_stack([x1, x2, x3, x4, x5, x6])
        
    # 7D data
    elif n_features == 7 and case == 1:
        start = [1, 1, 1, 1, 1, 1, 1]
        end = [2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1, 2 * np.pi - 1]
        samples_without_noise = generate_samples_on_line(start, end, n_samples)
        
    elif n_features == 7 and case == 2:
        x1 = np.random.uniform(1, 5, n_samples)
        x2 = np.random.uniform(2.5, 3, n_samples)
        x3 = np.sin(x1 * 2) + np.pi
        x4 = np.cos(x2 * 2) + np.pi
        x5 = np.sin(x4 * 2) + np.pi
        x6 = np.sin(x5 * 2) + np.pi
        x7 = (x1 + x2 + x3 + x4 + x5 + x6) / 6
        samples_without_noise = np.column_stack([x1, x2, x3, x4, x5, x6, x7])
        
    else:
        raise ValueError("Invalid combination of n_features and case.")
    
    # Add noise to the samples
    noise = np.random.normal(0, noise_level, samples_without_noise.shape)
    X = samples_without_noise + noise

    # Return noisy samples (X) and noise-free samples (samples_without_noise)
    samples = X
    return X, samples_without_noise

def generate_clustering_for_segments(n_features, case, n_samples, segments, noise_level):
    """
    Generate clustering data based on segments.
    
    Parameters:
        n_features (int): Number of features (must be greater than 3).
        case (int): Case number (1 for 3 segments, 2 for 5 segments).
        n_samples (int): Number of samples per segment.
        noise_level (float): Standard deviation of Gaussian noise.
    
    Returns:
        X (numpy.ndarray): Generated sample data with shape (num_segments * n_samples, n_features).
        yy (list): Labels corresponding to each segment.
    """
    if case == 1:
        num_segments = 3
    elif case == 2:
        num_segments = 5
    else:
        raise ValueError("case can be 1 or 2.")
    
    # Generate samples for each segment and add noise
    lines = [generate_samples_on_line(seg[0], seg[1], n_samples) for seg in segments]
    X = np.vstack([line + np.random.normal(0, noise_level, (n_samples, n_features)) for line in lines])
    yy = [label for label in range(num_segments) for _ in range(n_samples)]
    
    return X, yy

def data_generator_cl(n_features, case, n_samples, noise_level, random = False):
    """
    Generate samples for clustering experiments in dimensionality reduction.
    
    Parameters:
        n_features (int): Number of features/dimensions.
        case (int): Case number indicating the type of clustering data.
        n_samples (int): Number of samples per cluster.
        noise_level (float): Standard deviation of Gaussian noise.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        X (numpy.ndarray): Noisy data samples.
        yy (list): Cluster labels for each sample.
    """
    np.random.seed(42)
    if random:
        if n_features == 2 and case == 1:
            diagonal_points_1 = generate_samples_on_line([3*np.pi/4-1, 3*np.pi/4+1], [5*np.pi/4-1, 5*np.pi/4+1], n_samples)
            diagonal_points_2 = generate_samples_on_line([3*np.pi/4-1, 3*np.pi/4], [5*np.pi/4-1, 5*np.pi/4], n_samples)
            diagonal_points_3 = generate_samples_on_line([4, 2], [4, 3], n_samples)
            samples_without_noise = np.vstack((diagonal_points_1, diagonal_points_2, diagonal_points_3))
            noise = np.random.normal(0, noise_level, samples_without_noise.shape)
            X = samples_without_noise + noise
            yy = [0]*n_samples + [1]*n_samples + [2]*n_samples

        if n_features == 2 and case == 2:            
            # Generate sine wave samples
            x_values_sin = np.linspace(np.pi/4, 7 * np.pi/4, n_samples)
            y_values_sin = np.sin(x_values_sin * 2 + np.pi/4) / 3 + np.pi
            x_values_sin_1 = np.linspace(2*np.pi/4, 6 * np.pi/4, n_samples)
            y_values_sin_1 = np.sin(x_values_sin * 2 + np.pi/4) / 3 + np.pi
            x_values = np.linspace(3*np.pi/4, 5 * np.pi/4, n_samples)
            sin_samples_1 = np.column_stack((x_values_sin_1, y_values_sin_1))
            sin_samples_2 = np.column_stack((x_values, np.array([1] * len(y_values_sin)) * np.min(y_values_sin) - 1.5))
            sin_samples_3 = np.column_stack((x_values_sin, y_values_sin + 1.5))        
            samples_without_noise = np.vstack((sin_samples_1, sin_samples_2, sin_samples_3))
            noise = np.random.normal(0, noise_level, samples_without_noise.shape)
            X = samples_without_noise + noise
            yy = [0]*n_samples + [1]*n_samples + [2]*n_samples

        if n_features == 3 and case == 1:
            a1 = [2, 1, 2]
            a2 = [2.5, 2, 3]
            b1 = [2, 2, 3]
            b2 = [2, 3, 2]
            c1 = [3, 2, 1.5]
            c2 = [3, 2, 3.5]
            line1_start = np.array(a1)
            line1_end = np.array(a2)
            line2_start = np.array(b1)
            line2_end = np.array(b2)
            line3_start = np.array(c1)
            line3_end = np.array(c2)
            t = np.linspace(0, 1, n_samples)    
            line1 = line1_start + t[:, np.newaxis] * (line1_end - line1_start)
            line2 = line2_start + t[:, np.newaxis] * (line2_end - line2_start)
            line3 = line3_start + t[:, np.newaxis] * (line3_end - line3_start)    
            samples_without_noise = np.vstack((line1, line2, line3))
            noise = np.random.normal(0, noise_level, samples_without_noise.shape)
            X = samples_without_noise + noise
            yy = [0]*n_samples + [1]*n_samples + [2]*n_samples

        if n_features == 3 and case == 2:
            a, b = 1, 5
            S = total_arclength(a, b)
            uniform_arclengths = np.random.uniform(0, S, n_samples)
            x = np.array([arclength_to_x(s, a, b) for s in uniform_arclengths])
            x_1 = np.random.uniform(2, 4, n_samples)
            y = np.random.uniform(2.5, 3, n_samples)
            z = np.sin(x * 2) / 3 + np.pi
            samples_without_noise_1 = np.vstack((x, y, z)).T
            samples_without_noise_2 = np.vstack((x, y, z - 1)).T
            samples_without_noise_3 = np.vstack((x_1, y, np.array([1]*len(z)) * min(z) + 2)).T
            samples_without_noise = np.vstack((samples_without_noise_1, samples_without_noise_3, samples_without_noise_2))
            noise = np.random.normal(0, noise_level, samples_without_noise.shape)
            X = samples_without_noise + noise
            yy = [0]*n_samples + [1]*n_samples + [2]*n_samples


        elif n_features > 3:
            if random: 
                segments = generate_segments_with_min_distance(n_features = n_features, num_segments=3, min_distance=1.0)
                X, yy = generate_clustering_for_segments(n_features = n_features, case = case, n_samples = n_samples, segments = segments, noise_level = noise_level)
                
    else:
        X = np.loadtxt('Simulation_data/'+str(n_features)+'d_'+str(case)+'.csv', delimiter=',')
        if n_features > 3 and case == 2:
            yy = [0]*n_samples+[1]*n_samples+[2]*n_samples+[3]*n_samples+[4]*n_samples
        else:
            yy = [0]*n_samples+[1]*n_samples+[2]*n_samples
            
    return X, yy

