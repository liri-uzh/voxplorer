"""Dimensionality reduction algorithms.

This module contains functions for dimensionality reduction.
Apart from umap_reduction all methods are unsupervised and require only the data matrix `X` as input.
The umap_reduction method can take in an additional target vector `y` for supervised dimensionality reduction.
Requires the `umap-learn` package to be installed.
Requires the `scikit-learn` package to be installed.

Functions
---------
pca_reduction(X: np.ndarray, **kwargs)
    PCA dimensionality reduction.
umap_reduction(X: np.ndarray, y: np.ndarray = None, **kwargs)
    UMAP dimensionality reduction.
tsne_reduction(X: np.ndarray, **kwargs)
    t-SNE dimensionality reduction.
mds_reduction(X: np.ndarray, **kwargs)
    MDS dimensionality reduction.

Raises
------
ImportError if the `umap-learn` or `scikit-learn` packages are not installed.
ValueError if the dimensions of the input data are too small to reduce.
ValueError if the input data is not in the correct format.
"""

# Imports
import multiprocessing
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap


def _get_optimal_n_jobs():
    """Determines the optimal value for `n_jobs` using
    maximally 90% of resources.

    Returns:
    --------
    int: the optimal value for `n_jobs`
    """
    # get number of cores
    num_cores = multiprocessing.cpu_count()

    # calculate maximum number leaving 10% free
    num_jobs = max(1, int(num_cores * 0.9))

    return num_jobs


def pca_reduction(X: np.ndarray, **kwargs):
    """PCA dimensionality reduction.

    Parameters
    -----------
    X : np.ndarray
        Data to be reduced.
    **kwargs : dict
        Additional keyword arguments for PCA.

    Returns
    --------
    np.ndarray
        Reduced data.
    explained_variance : np.ndarray
        Explained variance.
    singluar_values : np.ndarray
        Singular values.
    """
    # Initialize PCA
    pca = PCA(**kwargs)
    # Fit and transform data
    X_pca = pca.fit_transform(X)
    # Return reduced data
    return X_pca, pca.explained_variance_, pca.singular_values_


def umap_reduction(X: np.ndarray, **kwargs):
    """UMAP dimensionality reduction.

    Parameters
    -----------
    X : np.ndarray
        Data to be reduced.
    **kwargs : dict
        Additional keyword arguments for UMAP.

    Returns
    --------
    np.ndarray
        Reduced data.
    """
    #  get n_jobs
    n_jobs = _get_optimal_n_jobs()

    # Initialize UMAP
    reducer = umap.UMAP(n_jobs=n_jobs, **kwargs)
    # Fit and transform data
    X_umap = reducer.fit_transform(X)
    # Return reduced data
    return X_umap


def tsne_reduction(X: np.ndarray, **kwargs):
    """t-SNE dimensionality reduction.

    Parameters
    -----------
    X : np.ndarray
        Data to be reduced.
    **kwargs : dict
        Additional keyword arguments for t-SNE.

    Returns
    --------
    np.ndarray
        Reduced data.
    """
    #  get n_jobs
    n_jobs = _get_optimal_n_jobs()

    # Initialize t-SNE
    tsne = TSNE(n_jobs=n_jobs, **kwargs)
    # Fit and transform data
    X_tsne = tsne.fit_transform(X)
    # Return reduced data
    return X_tsne


def mds_reduction(X: np.ndarray, **kwargs):
    """MDS dimensionality reduction.

    Parameters
    -----------
    X : np.ndarray
        Data to be reduced.
    **kwargs : dict
        Additional keyword arguments for MDS.

    Returns
    --------
    np.ndarray
        Reduced data.
    """
    #  get n_jobs
    n_jobs = _get_optimal_n_jobs()

    # Initialize MDS
    mds = MDS(n_jobs=n_jobs, **kwargs)
    # Fit and transform data
    X_mds = mds.fit_transform(X)
    # Return reduced data
    return X_mds
