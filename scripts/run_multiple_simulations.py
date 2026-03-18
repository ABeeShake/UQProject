import numpy as np
import time
import UQTools.generate as gen
import UQTools.simulate as sim
from UQTools.forecasters import fit_predict_regression
from UQTools.savi_variants import BootstrapSAVI_Combined, AdaptiveBettingSAVI, TamedONSSAVI
from UQTools.enbpi import enbpi_confidence_sequences

def evaluate_method(method, name, bootstrap_forecasts, Y_test, h):
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
    
    widths = U_seq - L_seq
    valid_widths = widths[np.isfinite(widths)]
    mean_width = np.mean(valid_widths) if len(valid_widths) > 0 else np.inf
    
    return mis_rate, mean_width, elapsed

if __name__ == "__main__":
    n_runs = 10
    n_samples = 400
    train_size = int(0.7 * n_samples)
    h = n_samples - train_size
    alpha = 0.1
    
    results_agg = {
        "EnbPI": {"miscoverages": [], "widths": [], "runtimes": []},
        "Combined Method": {"miscoverages": [], "widths": [], "runtimes": []},
        "Adaptive Betting": {"miscoverages": [], "widths": [], "runtimes": []},
        "Tamed ONS": {"miscoverages": [], "widths": [], "runtimes": []}
    }
    
    for run in range(n_runs):
        print(f"--- Run {run + 1}/{n_runs} ---")
        X_kwargs = {'loc': 0.0, 'scale': 1.0}
        noise_kwargs = {'loc': 0.0, 'scale': 1.0}
        func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))

        # We'll let np.random change the internal state across runs to ensure true randomization
        X = gen.generate_X(n_samples, d=1, distribution='normal', ar_coeffs=[0.7, -0.2], **X_kwargs)
        Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5], **noise_kwargs)

        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size].flatten(), Y[train_size:].flatten()

        B = 25
        X_bs, Y_bs = sim.block_bootstrap(X_train, Y_train, B=B)
        
        bootstrap_forecasts = np.zeros((B, h))
        for b in range(B):
            bootstrap_forecasts[b, :] = fit_predict_regression('rf', y_train=Y_bs[b], X_train=X_bs[b], h=h, X_test=X_test, 
                                            lags=3, n_estimators=50, random_state=42 + run*100 + b)
            
        start_time = time.time()
        L_seq_enbpi, U_seq_enbpi = enbpi_confidence_sequences(Y_test, bootstrap_forecasts, alpha=alpha)
        elapsed_enbpi = time.time() - start_time
        mis_rate_enbpi = np.mean((Y_test < L_seq_enbpi) | (Y_test > U_seq_enbpi))
        
        widths_enbpi = U_seq_enbpi - L_seq_enbpi
        valid_widths = widths_enbpi[np.isfinite(widths_enbpi)]
        mean_width_enbpi = np.mean(valid_widths) if len(valid_widths) > 0 else np.inf
        
        results_agg["EnbPI"]["miscoverages"].append(mis_rate_enbpi)
        results_agg["EnbPI"]["widths"].append(mean_width_enbpi)
        results_agg["EnbPI"]["runtimes"].append(elapsed_enbpi)
        
        methods = [
            ("Combined Method", BootstrapSAVI_Combined(alpha=alpha, window=50, gamma=0.5)),
            ("Adaptive Betting", AdaptiveBettingSAVI(alpha=alpha, window_size=50)),
            ("Tamed ONS", TamedONSSAVI(alpha=alpha, window_size=50))
        ]
        
        for name, method in methods:
            mis_rate, mean_width, elapsed = evaluate_method(method, name, bootstrap_forecasts, Y_test, h)
            results_agg[name]["miscoverages"].append(mis_rate)
            results_agg[name]["widths"].append(mean_width)
            results_agg[name]["runtimes"].append(elapsed)
            
    print("\n" + "="*80)
    print(f"SIMULATION RESULTS OVER {n_runs} RUNS".center(80))
    print("="*80)
    print(f"{'Method':<20} | {'Avg Miscoverage':<15} | {'Avg Width':<15} | {'Avg Runtime (s)':<15}")
    print("-" * 80)
    
    for name in ["EnbPI", "Combined Method", "Adaptive Betting", "Tamed ONS"]:
        avgs = {
            'miscoverage': np.mean(results_agg[name]["miscoverages"]),
            'width': np.mean(results_agg[name]["widths"]),
            'runtime': np.mean(results_agg[name]["runtimes"])
        }
        
        print(f"{name:<20} | {avgs['miscoverage']:<15.4f} | {avgs['width']:<15.4f} | {avgs['runtime']:<15.4f}")
        
    print("="*80)
