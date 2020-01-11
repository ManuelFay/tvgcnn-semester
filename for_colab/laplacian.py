import numpy as np
import scipy
import tensorflow as tf
from scipy.sparse.linalg import eigsh


def compute_laplacian_from_adjacency(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    W = W.astype(np.float32)
    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L


def rescale_laplacian(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def initialize_laplacian_tensor(W):
    L = compute_laplacian_from_adjacency(W, normalized=True)
    l, _ = eigsh(L, k=1)
    print(l)
    L = rescale_laplacian(L)
    L = L.tocoo()
    data = L.data.astype(np.float32)
    indices = np.empty((L.nnz, 2))
    indices[:, 0] = L.row
    indices[:, 1] = L.col
    L = tf.SparseTensor(indices, data, L.shape)
    L = tf.sparse.reorder(L)
    return L
