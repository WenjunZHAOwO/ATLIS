import numpy as np
import scipy

def construct_filtering_matrix(Lapmat, reg_lam):

    # Constructs a filtering matrix K = (I + reg_lam * Lapmat)^(-1) from a Laplacian matrix.
    n_samples = Lapmat.shape[0]

    if reg_lam == 0:
        return np.eye(n_samples)
    else:
        return np.linalg.pinv(np.eye(n_samples) + reg_lam*Lapmat)

def construct_LinOp_GraphPCA(Xmat, Kmat):
    """
    Constructs a LinearOperator representing the operation Xmat.T @ Kmat @ Xmat @ v.

    Args:
        Xmat (scipy.sparse.csr_matrix or numpy.ndarray):
            The feature matrix (n_samples x n_features).
        Kmat (scipy.sparse.csr_matrix or numpy.ndarray):
            The kernel or covariance-like matrix (n_samples x n_samples).

    Returns:
        scipy.sparse.linalg.LinearOperator:
            A linear operator that performs the specified matrix multiplication.
    """

    n_samples, n_features = Xmat.shape
    operator_shape = (n_features, n_features)

    def _matvec(v):
        """
        Defines the matrix-vector product for the linear operator:
        (Xmat.T @ Kmat @ Xmat) @ v
        """
        # Ensure v is a column vector if it's 1D for consistent matrix multiplication
        if v.ndim == 1:
            v_reshaped = v.reshape(-1, 1)
        else:
            v_reshaped = v

        # Multiplications:

        result = Xmat.T @ (Kmat @ (Xmat @ v_reshaped))

        # Return in the original shape of v if it was 1D
        if v.ndim == 1:
            return result.flatten()
        return result

    def _rmatvec(v):
        """
        Defines the matrix-vector product for the transpose of the linear operator:
        (Xmat.T @ Kmat @ Xmat).T @ v = Xmat.T @ Kmat.T @ Xmat @ v
        """
        return _matvec(v) # Valid if Kmat is symmetric
    
    # Create the LinearOperator
    myLinOp = scipy.sparse.linalg.LinearOperator(shape=operator_shape, matvec=_matvec, rmatvec=_rmatvec)

    return myLinOp

def GraphPCA(Xmat, Kmat, k_proj, method='LinOp'):
        
    """
    Performs Graph-regularized Principal Component Analysis (GraphPCA).

    This function computes the principal components (eigenvectors) and corresponding
    eigenvalues for a graph-regularized covariance matrix, M = Xmat.T @ Kmat @ Xmat.
    It supports different computation methods for M.

    Parameters
    ----------
    Xmat : numpy.ndarray or scipy.sparse.csr_matrix
        The data matrix, with shape (n_samples, n_features). Each row represents
        a sample, and each column represents a feature.
    Kmat : numpy.ndarray or scipy.sparse.csr_matrix
        A "filtering" matrix, with shape (n_samples, n_samples).
        This should be computed as (I + lambda * L)^(-1), where I is the identity matrix,
        lambda is a regularization parameter, and L is the graph Laplacian.
    k_proj : int
        The desired rank of the decomposition, i.e., the number of top
        eigenvectors and eigenvalues to compute.
    method : str, optional
        Specifies the method for computing the principal components.
        - 'LinOp': Uses `scipy.sparse.linalg.LinearOperator` to implicitly
          represent `M = Xmat.T @ Kmat @ Xmat`, which is passed to `eigsh`.
          This is memory-efficient for large `n_features`.
        - 'svd': Computes `scipy.linalg.sqrtm(Kmat)` and then uses `svds`
          on `Xmat.T @ sqrtm(Kmat)`. This approach works on the symmetric square root
          and can be robust.
        - Any other string (e.g., 'explicit'): Explicitly computes `M = Xmat.T @ Kmat @ Xmat`
          and then passes the full matrix `M` to `eigsh`. This can be memory-intensive
          for large `n_features`.
        Defaults to 'LinOp'.

    Returns
    -------
    dict
        A dictionary containing the results:
        - 'Zmat' : numpy.ndarray
            The projected data matrix (n_samples, k_proj). This is computed as
            `Kmat @ (Xmat @ Wmat)`.
        - 'Wmat' : numpy.ndarray
            The matrix of eigenvectors (principal components) corresponding to
            the `k_proj` largest eigenvalues, with shape (n_features, k_proj).
        - 'eigenvalues' : numpy.ndarray
            A 1D array of the `k_proj` largest eigenvalues of the matrix `M`,
            sorted in descending order.

    """
    if method == 'svd':
        M = Xmat.T @ scipy.linalg.sqrtm(Kmat)
        Wmat, singvalues, _ = scipy.sparse.linalg.svds(M, k=k_proj, which='LM', return_singular_vectors=True)
        eigenvalues = singvalues**2

    else:
        if method == 'LinOp':
            M = construct_LinOp_GraphPCA(Xmat, Kmat)
        else:
            M = Xmat.T @ Kmat @ Xmat
        
        eigenvalues, Wmat = scipy.sparse.linalg.eigsh(M, k=k_proj, which='LM', return_eigenvectors=True)
        
    Zmat = Kmat @ (Xmat @ Wmat)
    sorting_indices = np.argsort(eigenvalues)[::-1]
    return {"Zmat": Zmat[:,sorting_indices], "Wmat": Wmat[:, sorting_indices], "eigenvalues": eigenvalues[sorting_indices]}