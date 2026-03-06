import numpy as np
import matplotlib.pyplot as plt
import UQTools.generate as gen
import UQTools.simulate as sim
from UQTools.savi import savi_confidence_sequences

if __name__ == "__main__":
    
    #Simulation Settings
    n_samples = 400
    d = 1
    X_kwargs = {'loc': 0.0, 'scale': 1.0}

    noise_kwargs = {'loc': 0.0, 'scale': 1.0}
    func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))

    train_size = int(0.7 * n_samples)
    h = n_samples - train_size

    #Generate Covariates
    X = gen.generate_X(n_samples, d=1, distribution='normal', ar_coeffs=[0.7, -0.2], **X_kwargs)
    Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5], **noise_kwargs)

    #Train-Test Split
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size].flatten(), Y[train_size:].flatten()

    arima_kwargs = {
        'seasonal': True,
        'season_length': 12,
        'max_p': 2,
        'max_d': 1,
        'max_q': 2
    }

    # Block Bootstrap resampling (X^b, Y^b) for training
    B = 25
    block_size = 10
    print(f"\nGenerating {B} block-bootstrap samples from (X_train, Y_train) ...")
    
    X_bs, Y_bs = sim.block_bootstrap(X_train, Y_train, B=B, block_size=block_size, setting=1)
    
    # Store forecasts from each bootstrap sample
    bootstrap_forecasts = np.zeros((B, h))
    
    print("Fitting ARIMA to each bootstrap sample...")
    for b in range(B):
        if b % 5 == 0:
            print(f"  Processed {b}/{B} samples...")
            
        X_train_b = X_bs[b]
        Y_train_b = Y_bs[b]
        
        
        from UQTools.forecasters import fit_predict_regression
        y_hat_b = fit_predict_regression('rf', y_train=Y_train_b, X_train=X_train_b, h=h, X_test=X_test, 
                                        lags=3, n_estimators=50, random_state=42)
        bootstrap_forecasts[b, :] = y_hat_b

    print("Running SAVI Confidence Sequences...")
    alpha = 0.1 # 90% confidence
    L_seq, U_seq = savi_confidence_sequences(y_true=Y_test, mu_preds_bootstrapped=bootstrap_forecasts, alpha=alpha)

    # Calculate Miscoverage Rate
    miscoverage_flags = (Y_test < L_seq) | (Y_test > U_seq)
    miscoverage_rate = np.mean(miscoverage_flags)

    print(f"Target Alpha (Max Miscoverage): {alpha}")
    print(f"Empirical Miscoverage Rate: {miscoverage_rate:.4f}")

    # Plot
    start = 0
    end = n_samples
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(train_size, n_samples)[start:end], Y_test[start:end], color='blue', label='True Test Data')
    plt.plot(np.arange(train_size, n_samples)[start:end], np.mean(bootstrap_forecasts, axis=0)[start:end], color='orange', label='Bootstrap Mean')
    
    plt.plot(np.arange(train_size, n_samples)[start:end], U_seq[start:end], color='green', linestyle='--', label='SAVI Upper Bound')
    plt.plot(np.arange(train_size, n_samples)[start:end], L_seq[start:end], color='red', linestyle='--', label='SAVI Lower Bound')
    
    # Cap bounds for visualization
    valid_u = U_seq[np.isfinite(U_seq)]
    valid_l = L_seq[np.isfinite(L_seq)]
    if len(valid_u) > 0 and len(valid_l) > 0:
        max_y = max(np.max(Y_test), np.max(valid_u)) + 2
        min_y = min(np.min(Y_test), np.min(valid_l)) - 2
        plt.ylim(min_y, max_y)
        
    plt.fill_between(np.arange(train_size, n_samples), L_seq, U_seq, color='gray', alpha=0.2)

    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.title(f'SAVI Confidence Sequences (alpha={alpha})\nMiscoverage Rate: {miscoverage_rate:.4f}')
    plt.legend()
    plt.savefig('./imgs/savi_forecast.png')
    plt.close()

    print("Done! Plot saved to ./imgs/savi_forecast.png")
