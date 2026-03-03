import numpy as np
from statsforecast.models import AutoARIMA
def block_bootstrap(X, Y, B=1000, block_size=10, setting=1):
    """
    X: t x d
    Y: t x 1
    B: number of bootstrap samples
    block_size: length of blocks to sample

    Resamples a time series {(x_s,y_s)}_{s=1}^t using block bootstrap

    * Setting 1: Resample blocks of pairs (x_s,y_s) with replacement
    * Setting 2: Resample blocks of x_s and y_s independently with replacement
    """
    t = X.shape[0]
    d = X.shape[1] if X.ndim > 1 else 1
    num_blocks = int(np.ceil(t / block_size))
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    Y_dim = 1 if Y.ndim == 1 else Y.shape[1]
    X_bs = np.zeros((B, t, d))
    Y_bs = np.zeros((B, t, Y_dim))
    
    Y_reshaped = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        
    max_start_idx = t - block_size + 1
    if max_start_idx <= 0:
        raise ValueError(f"Time series length {t} must be >= block_size {block_size}")
        
    for b in range(B):
        if setting == 1:
            idx = np.random.randint(0, max_start_idx, size=num_blocks)
            x_blocks = [X[i:i+block_size] for i in idx]
            y_blocks = [Y_reshaped[i:i+block_size] for i in idx]
            
            X_bs[b] = np.vstack(x_blocks)[:t]
            Y_bs[b] = np.vstack(y_blocks)[:t]
            
        elif setting == 2:
            idx_x = np.random.randint(0, max_start_idx, size=num_blocks)
            idx_y = np.random.randint(0, max_start_idx, size=num_blocks)
            
            x_blocks = [X[i:i+block_size] for i in idx_x]
            y_blocks = [Y_reshaped[i:i+block_size] for i in idx_y]
            
            X_bs[b] = np.vstack(x_blocks)[:t]
            Y_bs[b] = np.vstack(y_blocks)[:t]
        else:
            raise ValueError("Setting must be 1 or 2")
            
    return X_bs, Y_bs


def find_block_length(X,Y):
    """
    X: t x d
    Y: t x 1

    Find optimal block length for block bootstrap
    * Idea 1: minimize MSE between original and bootstrapped series
    * Idea 2: use autocorrelation cutoff
    """
    pass
