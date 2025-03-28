from utils import find_nbr
import numpy as np
from scipy.sparse.linalg import svds

def psm(sample, h, maxiter, rho, e_n, alpha=1):
    """
    Perform a point set manifold (PSM) processing algorithm to adjust the sample points.
    
    Parameters:
        sample (numpy.ndarray): Input data samples (each row is a sample).
        h (float): Neighborhood radius for finding neighbors.
        maxiter (int): Maximum number of iterations for each point.
        rho (float): Convergence threshold (if the change is smaller than rho, stop the inner loop).
        e_n (int): Controls the number of singular vectors to retain, which in turn controls the dimension.
        alpha (float): Learning rate for the update steps.
        
    Returns:
        numpy.ndarray: Updated sample points after the PSM algorithm.
    """
    # Transpose the sample matrix so that each column is a sample point
    X = sample.T
    N = X.shape[1]  # Total number of sample points

    # Loop over a maximum of maxiter iterations for each sample point
    for iter in range(maxiter):
        # Process each sample point (column in X)
        for ii in range(N):
            # Get the current point x
            x = X[:, ii]
            pp = 0  # Initialize inner iteration counter
            
            # Inner loop to update x until convergence or a maximum of 100 inner iterations
            while pp < 100:
                # Find the neighbors of the current point x within radius h
                # Note: find_nbr expects the first argument as a point and the second as the collection of points.
                X0 = find_nbr(x, sample.T, h)
                
                # Compute the mean of the neighbors along each dimension (xbar)
                xbar = np.mean(X0, axis=1)
                
                # If the number of neighbor points is greater than the dimension, perform SVD on the centered data
                if X0.shape[1] > X0.shape[0]:
                    # Compute the singular value decomposition on the centered neighbor data
                    # Retain e_n smallest singular vectors ('SM' indicates smallest magnitude)
                    u, _ , _ = svds(X0 - xbar.reshape(-1, 1), k=e_n, which='SM')
                    # Project the difference (xbar - x) onto the subspace spanned by the singular vectors
                    nv = (u @ u.T) @ (xbar - x)
                else:
                    # If not enough neighbors, simply use the difference between the mean and current point
                    nv = xbar - x
                
                # Update the current point using the computed adjustment vector nv scaled by the learning rate alpha
                xnew = x + alpha * nv
                
                # Check for convergence: if the change is below the threshold rho, break the inner loop
                if np.linalg.norm(xnew - x) < rho:
                    break
                    
                # Update x to the new value and increment the inner iteration counter
                x = xnew
                pp += 1
            
            # Normalize pairs of coordinates (assuming they represent an angle in cosine-sine form)
            # Iterate over each pair (step size 2) in the current point vector
            for dd in range(0, sample.T.shape[0], 2):
                # Normalize the two components so that the resulting vector has unit norm
                X[dd:dd+2, ii] = x[dd:dd+2] / np.linalg.norm(x[dd:dd+2])
    
    # Return the processed points transposed back to the original format (each row is a sample)
    return X.T
