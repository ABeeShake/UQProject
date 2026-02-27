import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import UQTools.generate as gen
import UQTools.simulate as sim

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

if __name__ == "__main__":
    
    #Simulation Settings
    n_samples = 1000
    d = 1
    distribution = 'normal'
    X_kwargs = {'loc': 0.0, 'scale': 1.0}

    noise_distribution = 'normal'
    noise_kwargs = {'loc': 0.0, 'scale': 1.0}
    func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))

    train_size = int(0.8 * n_samples)
    h = n_samples - train_size

    #Generate Covariates
    X = gen.generate_X(n_samples, d=1, distribution='normal', ar_coeffs=[0.7, -0.2], **X_kwargs)
    print(f"X shape: {X.shape}")

    #Generate Target
    Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5], **noise_kwargs)
    print(f"Y shape: {Y.shape}")

    #Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Data')
    plt.savefig('./imgs/data.png')
    plt.close()

    #Train-Test Split
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size].flatten(), Y[train_size:].flatten()

    # AutoARIMA kwargs to handle complex patterns
    arima_kwargs = {
        'seasonal': True,
        'season_length': 12,
        'max_p': 5,
        'max_d': 2,
        'max_q': 5
    }

    # Point Predictor for T+h
    y_hat = sim.fit_predict_arima(y_train=Y_train, X_train=X_train, h=h, X_test=X_test, **arima_kwargs)
    
    print(f"Predicted y_hat (first {min(5, h)} steps): {y_hat[:min(5, h)]}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(train_size), Y_train, 
            alpha=0.5, label='Training Data', color='blue')
    plt.plot(np.arange(train_size,n_samples), Y_test, 
            alpha=0.5, label='Test Data', color='blue',linestyle='--')
    plt.plot(np.arange(train_size,n_samples), y_hat, 
            color='red', label='ARIMA Forecast')
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.title('Generated Data and ARIMA Forecast')
    plt.legend()
    plt.savefig('./imgs/data_and_forecast.png')
    plt.close()

    # Block Bootstrap resampling (X^b, Y^b)
    B = 100
    block_size = 10  # Default or can be tuned
    print(f"\nGenerating {B} block-bootstrap samples from (X, Y) ...")
    
    # Passing the full series to generate replicate futures implicitly
    X_bs, Y_bs = sim.block_bootstrap(X, Y, B=B, block_size=block_size, setting=1)
    
    print(f"X_bs shape: {X_bs.shape}")
    print(f"Y_bs shape: {Y_bs.shape}")
    
    # Store forecasts from each bootstrap sample
    bootstrap_forecasts = np.zeros((B, h))
    
    print("Fitting ARIMA to each bootstrap sample...")
    for b in range(B):
        if b % 10 == 0:
            print(f"  Processed {b}/{B} samples...")
            
        X_train_b = X_bs[b, :train_size]
        Y_train_b = Y_bs[b, :train_size]
        X_test_b = X_bs[b, train_size:]
        
        # We model the relation and forecast h steps into the future.
        y_hat_b = sim.fit_predict_arima(y_train=Y_train_b, X_train=X_train_b, h=h, X_test=X_test_b, **arima_kwargs)
        bootstrap_forecasts[b, :] = y_hat_b

    # Compute bounds and quantiles
    q_lower = np.quantile(bootstrap_forecasts, 0.025, axis=0)
    q_upper = np.quantile(bootstrap_forecasts, 0.975, axis=0)
    y_hat_mean = np.mean(bootstrap_forecasts, axis=0)
    
    # Plot Bootstrapped Intervals vs Actuals
    plt.figure(figsize=(10, 6))
    
    # Optional: plot the point forecast from before as a reference
    # plt.plot(np.arange(train_size, n_samples), y_hat, color='red', label='Point Forecast', linestyle='-.')
    
    plt.plot(np.arange(train_size, n_samples), y_hat_mean, color='orange', label='Bootstrap Mean Forecast')
    plt.fill_between(np.arange(train_size, n_samples), q_lower, q_upper, color='orange', alpha=0.3, label='95% Confidence Interval')
    
    plt.plot(np.arange(train_size, n_samples), Y_test, color='blue', label='True Test Data')
    
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.title('ARIMA Forecast with 95% Bootstrap Confidence Intervals')
    plt.legend()
    plt.savefig('./imgs/bootstrap_forecast.png')
    plt.close()
    
    print("Done! Plot saved to ./imgs/bootstrap_forecast.png")