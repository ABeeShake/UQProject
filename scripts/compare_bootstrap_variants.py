import numpy as np
import matplotlib.pyplot as plt
import time
import UQTools.generate as gen
import UQTools.simulate as sim
from UQTools.forecasters import fit_predict_regression
from UQTools.savi import BootstrapSAVI
from UQTools.savi_variants import BootstrapSAVI_Combined, AdaptiveBettingSAVI, TamedONSSAVI
from UQTools.enbpi import enbpi_confidence_sequences

def evaluate_method(method, name, bootstrap_forecasts, Y_test, h):
    print(f"  Evaluating {name}...")
    L_seq = np.zeros(h)
    U_seq = np.zeros(h)
    
    start_time = time.time()
    for t in range(h):
        ci = method.confidence_interval(bootstrap_forecasts[:, t])
        if ci is not None:
            L_seq[t], U_seq[t] = ci
        else:
            L_seq[t], U_seq[t] = np.inf, -np.inf
            
        method.update(Y_test[t], bootstrap_forecasts[:, t])
        
    elapsed = time.time() - start_time
    
    mis_rate = np.mean((Y_test < L_seq) | (Y_test > U_seq))
    mean_width = np.mean(U_seq - L_seq)
    
    print(f"    Miscoverage: {mis_rate:.4f} | Avg Width: {mean_width:.4f} | Runtime: {elapsed:.2f}s")
    return L_seq, U_seq, mis_rate, mean_width, elapsed

if __name__ == "__main__":
    n_samples = 400
    train_size = int(0.7 * n_samples)
    h = n_samples - train_size

    X_kwargs = {'loc': 0.0, 'scale': 1.0}
    noise_kwargs = {'loc': 0.0, 'scale': 1.0}
    func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))

    X = gen.generate_X(n_samples, d=1, distribution='normal', ar_coeffs=[0.7, -0.2], **X_kwargs)
    Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5], **noise_kwargs)

    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size].flatten(), Y[train_size:].flatten()

    B = 25
    block_size = 10
    print(f"\nGenerating {B} block-bootstrap samples...")
    X_bs, Y_bs = sim.block_bootstrap(X_train, Y_train, B=B, block_size=block_size, setting=1)
    
    bootstrap_forecasts = np.zeros((B, h))
    for b in range(B):
        bootstrap_forecasts[b, :] = fit_predict_regression('rf', y_train=Y_bs[b], X_train=X_bs[b], h=h, X_test=X_test, 
                                        lags=3, n_estimators=50, random_state=42)

    alpha = 0.1
    print("\nRunning Confidence Sequences...")
    
    methods = [
        ("Combined Method", BootstrapSAVI_Combined(alpha=alpha, window=50, gamma=0.5)),
        ("Adaptive Betting", AdaptiveBettingSAVI(alpha=alpha, window_size=50)),
        ("Tamed ONS", TamedONSSAVI(alpha=alpha, window_size=50))
    ]
    
    results = []
    
    # Run EnbPI benchmark separately because it is function-based
    print(f"  Evaluating EnbPI...")
    start_time = time.time()
    L_seq_enbpi, U_seq_enbpi = enbpi_confidence_sequences(Y_test, bootstrap_forecasts, alpha=alpha)
    elapsed_enbpi = time.time() - start_time
    mis_rate_enbpi = np.mean((Y_test < L_seq_enbpi) | (Y_test > U_seq_enbpi))
    mean_width_enbpi = np.mean(U_seq_enbpi - L_seq_enbpi)
    print(f"    Miscoverage: {mis_rate_enbpi:.4f} | Avg Width: {mean_width_enbpi:.4f} | Runtime: {elapsed_enbpi:.2f}s")
    results.append(("EnbPI", L_seq_enbpi, U_seq_enbpi, mis_rate_enbpi, mean_width_enbpi, elapsed_enbpi))
    
    for name, method in methods:
        L_seq, U_seq, mis_rate, mean_width, elapsed = evaluate_method(method, name, bootstrap_forecasts, Y_test, h)
        results.append((name, L_seq, U_seq, mis_rate, mean_width, elapsed))
        
    # Plotting
    mu_preds = np.mean(bootstrap_forecasts, axis=0)
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)), sharex=True)
    
    for idx, (name, L_seq, U_seq, mis_rate, mean_width, elapsed) in enumerate(results):
        ax = axes[idx]
        ax.plot(Y_test, color='blue', label='True Test Data')
        ax.plot(mu_preds, color='orange', label='Bootstrap Mean')
        
        ax.plot(U_seq, color='green', linestyle='--', label=f'Upper Bound')
        ax.plot(L_seq, color='red', linestyle='--', label=f'Lower Bound')
        ax.fill_between(np.arange(len(Y_test)), L_seq, U_seq, color='gray', alpha=0.2)
        
        ax.set_title(f'{name} | Miscoverage: {mis_rate:.4f} (Target {alpha}) | Width: {mean_width:.2f} | Time: {elapsed:.2f}s')
        ax.legend(loc='lower left')
        
    plt.tight_layout()
    plt.savefig('./imgs/variance_reduction_comparison.png')
    plt.close()
    print("\nPlot saved to ./imgs/variance_reduction_comparison.png")
