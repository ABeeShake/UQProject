import numpy as np

def generate_X(n_samples, d=1, distribution='normal', ar_coeffs=None, **kwargs):
    """
    Generate covariates X_t from a specified distribution, optionally with autoregressive dynamics.
    
    Parameters:
    n_samples (int): Number of time steps (T)
    d (int): Dimension of covariates
    distribution (str or callable): Distribution name ('normal', 'uniform') or a callable.
    ar_coeffs (list or np.ndarray): Coefficients for autoregressive process [phi_1, ..., phi_p].
    **kwargs: Arguments for the distribution (can be scalar or time-varying arrays of shape (n_samples, d))
    
    Returns:
    np.ndarray: covariates X of shape (n_samples, d)
    """
    if callable(distribution):
        base_X = distribution(n_samples, d, **kwargs)
    elif distribution == 'normal':
        loc = kwargs.get('loc', 0.0)
        scale = kwargs.get('scale', 1.0)
        base_X = np.random.normal(loc, scale, size=(n_samples, d))
    elif distribution == 'uniform':
        low = kwargs.get('low', 0.0)
        high = kwargs.get('high', 1.0)
        base_X = np.random.uniform(low, high, size=(n_samples, d))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
        
    if ar_coeffs is not None:
        ar_coeffs = np.array(ar_coeffs)
        p = len(ar_coeffs)
        X = np.zeros_like(base_X)
        X[:p] = base_X[:p]
        for t in range(p, n_samples):
            ar_part = np.sum([ar_coeffs[i] * X[t-1-i] for i in range(p)], axis=0)
            X[t] = ar_part + base_X[t]
        return X

    return base_X


def generate_y(X, func, noise_distribution='normal', ar_coeffs=None, **kwargs):
    """
    Generate y_t as a specified function of X_t with some random noise, optionally with autoregressive dynamics on y.
    
    Parameters:
    X (np.ndarray): Covariates of shape (n_samples, d)
    func (callable): Function taking X and returning true y (base signal without noise)
    noise_distribution (str): Distribution of the random noise ('normal', 'uniform', or None)
    ar_coeffs (list or np.ndarray): Coefficients for AR process on y [phi_1, ..., phi_p].
    **kwargs: Arguments for the noise distribution
    
    Returns:
    np.ndarray: target variable y of shape (n_samples, 1)
    """
    n_samples = X.shape[0]
    
    # Generate the base signal
    y_base = np.array(func(X), dtype=float)
    if y_base.ndim == 1:
        y_base = y_base.reshape(-1, 1)
        
    # Add random noise
    if noise_distribution == 'normal':
        loc = kwargs.get('loc', 0.0)
        scale = kwargs.get('scale', 1.0)
        noise = np.random.normal(loc, scale, size=(n_samples, 1))
    elif noise_distribution == 'uniform':
        low = kwargs.get('low', -1.0)
        high = kwargs.get('high', 1.0)
        noise = np.random.uniform(low, high, size=(n_samples, 1))
    elif noise_distribution is None:
        noise = np.zeros((n_samples, 1))
    else:
        raise ValueError(f"Unknown noise distribution: {noise_distribution}")
        
    if ar_coeffs is not None:
        ar_coeffs = np.array(ar_coeffs)
        p = len(ar_coeffs)
        y = np.zeros_like(y_base)
        y[:p] = y_base[:p] + noise[:p]
        for t in range(p, n_samples):
            ar_part = np.sum([ar_coeffs[i] * y[t-1-i] for i in range(p)], axis=0)
            y[t] = y_base[t] + noise[t] + ar_part
        return y
        
    return y_base + noise