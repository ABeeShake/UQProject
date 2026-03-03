import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import UQTools.generate as gen
import UQTools.simulate as sim

def compute_stats(y):
    y = y.flatten()
    mean = np.mean(y)
    var = np.var(y)
    # Lag-1 Autocorrelation
    if len(y) > 1:
        corr = np.corrcoef(y[:-1], y[1:])[0, 1]
    else:
        corr = 0
    return mean, var, corr

if __name__ == "__main__":
    n_samples = 400
    train_size = int(0.7 * n_samples)
    
    X_kwargs = {'loc': 0.0, 'scale': 1.0}
    noise_kwargs = {'loc': 0.0, 'scale': 1.0}
    func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))

    # Generate Covariates and Target
    X = gen.generate_X(n_samples, d=1, distribution='normal', ar_coeffs=[0.7, -0.2], **X_kwargs)
    Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5], **noise_kwargs)

    X_train, Y_train = X[:train_size], Y[:train_size].flatten()

    B = 5
    block_size = 10
    print(f"Original Time Series Stats (n={len(Y_train)}):")
    orig_mean, orig_var, orig_corr = compute_stats(Y_train)
    print(f"  Mean: {orig_mean:.4f}, Variance: {orig_var:.4f}, Lag-1 ACF: {orig_corr:.4f}")

    # Generate block-bootstrap samples
    X_bs, Y_bs = sim.block_bootstrap(X_train, Y_train, B=B, block_size=block_size, setting=1)

    plt.figure(figsize=(12, 8))
    
    # Plot original
    plt.subplot(B+1, 1, 1)
    plt.plot(Y_train, color='blue', label='Original')
    plt.title('Original Target Series (Y_train)')
    plt.xlim(0, len(Y_train))
    plt.legend()
    
    for b in range(B):
        y_b = Y_bs[b].flatten()
        b_mean, b_var, b_corr = compute_stats(y_b)
        print(f"Bootstrap Sample {b+1} Stats:")
        print(f"  Mean: {b_mean:.4f}, Variance: {b_var:.4f}, Lag-1 ACF: {b_corr:.4f}")
        
        plt.subplot(B+1, 1, b+2)
        plt.plot(y_b, color='orange', label=f'Bootstrap {b+1}')
        plt.title(f'Sample {b+1}')
        plt.xlim(0, len(y_b))
        plt.legend()

    plt.tight_layout()
    plt.savefig('./imgs/bootstrap_analysis.png')
    plt.close()
    print("\nPlot saved to ./imgs/bootstrap_analysis.png")
