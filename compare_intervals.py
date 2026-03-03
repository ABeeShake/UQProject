import numpy as np
import matplotlib.pyplot as plt
import UQTools.generate as gen
import UQTools.simulate as sim

from UQTools.intervals import savi_confidence_sequences,wsr_confidence_sequences,enbpi_confidence_sequences
from UQTools.forecasters import fit_predict_regression

if __name__ == "__main__":
    
    #Simulation Settings
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

    # Block Bootstrap resampling (X^b, Y^b) for training
    B = 10
    block_size = 10
    print(f"\nGenerating {B} block-bootstrap samples from (X_train, Y_train) ...")
    X_bs, Y_bs = sim.block_bootstrap(X_train, Y_train, B=B, block_size=block_size, setting=1)
    
    bootstrap_forecasts = np.zeros((B, h))
    
    print("Fitting Random Forest to each bootstrap sample...")
    for b in range(B):
        if b % 5 == 0:
            print(f"  Processed {b}/{B} samples...")
        y_hat_b = fit_predict_regression('rf', y_train=Y_bs[b], X_train=X_bs[b], h=h, X_test=X_test, 
                                        lags=3, n_estimators=50, random_state=42)
        bootstrap_forecasts[b, :] = y_hat_b

    alpha = 0.1
    print("Running Confidence Sequences...")
    
    mu_preds = np.mean(bootstrap_forecasts, axis=0)

    # 1. SAVI
    print("  Evaluating SAVI...")
    L_savi, U_savi = savi_confidence_sequences(y_true=Y_test, mu_preds_bootstrapped=bootstrap_forecasts, alpha=alpha)
    mis_savi = np.mean((Y_test < L_savi) | (Y_test > U_savi))
    
    # 2. WSR
    print("  Evaluating WSR...")
    L_wsr, U_wsr = wsr_confidence_sequences(y_true=Y_test, mu_preds=mu_preds, alpha=alpha)
    mis_wsr = np.mean((Y_test < L_wsr) | (Y_test > U_wsr))
    
    # 3. EnbPI
    print("  Evaluating EnbPI...")
    L_enb, U_enb = enbpi_confidence_sequences(y_true=Y_test, mu_preds_bootstrapped=bootstrap_forecasts, alpha=alpha)
    mis_enb = np.mean((Y_test < L_enb) | (Y_test > U_enb))

    print(f"\n--- Miscoverage Rates (Target: {alpha}) ---")
    print(f"SAVI:  {mis_savi:.4f}")
    print(f"WSR:   {mis_wsr:.4f}")
    print(f"EnbPI: {mis_enb:.4f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    methods = [
        ("SAVI", L_savi, U_savi, mis_savi),
        ("WSR (Variance Betting)", L_wsr, U_wsr, mis_wsr),
        ("EnbPI (Conformal)", L_enb, U_enb, mis_enb)
    ]
    
    for idx, (name, L_seq, U_seq, mis_rate) in enumerate(methods):
        ax = axes[idx]
        ax.plot(Y_test, color='blue', label='True Test Data')
        ax.plot(mu_preds, color='orange', label='Bootstrap Mean')
        
        ax.plot(U_seq, color='green', linestyle='--', label=f'{name} Upper')
        ax.plot(L_seq, color='red', linestyle='--', label=f'{name} Lower')
        
        ax.fill_between(np.arange(len(Y_test)), L_seq, U_seq, color='gray', alpha=0.2)
        
        ax.set_title(f'{name} CS (Miscoverage: {mis_rate:.4f})')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('./imgs/compare_intervals.png')
    plt.close()
    print("\nPlot saved to ./imgs/compare_intervals.png")
