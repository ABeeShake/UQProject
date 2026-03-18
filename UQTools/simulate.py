import numpy as np

def find_block_length(series):
    """
    Implements a simplified Politis and White (2004) spectral plug-in rule.
    The optimal block length b* is proportional to (G / D)^(1/3) * N^(1/3).
    """
    n = len(series)
    if n < 20: return 1
    
    # 1. Compute Autocorrelations (ACF)
    # We look for the first lag where ACF is effectively zero to estimate 'G'
    def get_acf(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag < n else 0

    lags = min(n // 2, 100)
    acf_vals = [get_acf(series, k) for k in range(1, lags)]
    
    # Find m: the smallest lag after which ACFs are 'insignificant'
    # Threshold per Politis & White: 2 * sqrt(log10(n) / n)
    threshold = 2 * np.sqrt(np.log10(n) / n)
    m = 1
    for i, rho in enumerate(acf_vals):
        if abs(rho) < threshold:
            m = i + 1
            break
            
    # 2. Compute the 'G' parameter (Sum of k * R(k))
    # This represents the strength of the dependency
    g = sum(k * acf_vals[k-1] for k in range(1, m))
    
    # 3. Calculate optimal block length b_star
    # For Moving Block Bootstrap, the constant is approx 1.14 ( (3/4)^(1/3) )
    b_star = 1.14 * (abs(g)**(2/3)) * (n**(1/3))
    
    # Clamp to reasonable limits
    return int(np.clip(b_star, 1, n // 2))

def block_bootstrap(X, Y, B=1000):
    """
    Updated: Only Setting 1 (Paired Resampling) is supported.
    Automatically finds the optimal block size based on Y-series dependency.
    """
    t = X.shape[0]
    d = X.shape[1] if X.ndim > 1 else 1
    
    # 1. Determine block size automatically
    block_size = find_block_length(Y.flatten())
    num_blocks = int(np.ceil(t / block_size))
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    Y_dim = 1 if Y.ndim == 1 else Y.shape[1]
    X_bs = np.zeros((B, t, d))
    Y_bs = np.zeros((B, t, Y_dim))
    
    Y_reshaped = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    max_start_idx = t - block_size + 1
        
    # 2. Perform Paired Resampling (Setting 1)
    for b in range(B):
        # Sample block starting indices with replacement
        idx = np.random.randint(0, max_start_idx, size=num_blocks)
        
        x_blocks = [X[i:i+block_size] for i in idx]
        y_blocks = [Y_reshaped[i:i+block_size] for i in idx]
        
        # Concatenate and trim to original length t
        X_bs[b] = np.vstack(x_blocks)[:t]
        Y_bs[b] = np.vstack(y_blocks)[:t]
            
    return X_bs, Y_bs