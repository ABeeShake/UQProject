import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import UQTools.generate as gen
import UQTools.simulate as sim
from UQTools.forecasters import fit_predict_regression

if __name__ == "__main__":
    n_samples = 400
    train_size = int(0.7 * n_samples)
    h_test = n_samples - train_size
    
    X_kwargs = {'loc': 0.0, 'scale': 1.0}
    noise_kwargs = {'loc': 0.0, 'scale': 1.0}
    func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))

    X = gen.generate_X(n_samples, d=1, distribution='normal', ar_coeffs=[0.7, -0.2], **X_kwargs)
    Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5], **noise_kwargs)

    X_train, Y_train = X[:train_size], Y[:train_size].flatten()
    X_test, Y_test = X[train_size:], Y[train_size:].flatten()

    lags = 3
    
    models = {
        'ARIMA': lambda y, x, h, xt: fit_predict_regression('arima', y, x, h, xt, seasonal=True, season_length=12, max_p=2, max_d=1, max_q=2),
        'Ridge': lambda y, x, h, xt: fit_predict_regression('ridge', y, x, h, xt, lags=lags, alpha=1.0),
        'Random Forest': lambda y, x, h, xt: fit_predict_regression('rf', y, x, h, xt, lags=lags, n_estimators=50, random_state=42),
        'SVR': lambda y, x, h, xt: fit_predict_regression('svr', y, x, h, xt, lags=lags, kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    predictions = {name: [] for name in models.keys()}
    
    print("Evaluating models online (1-step ahead)...")
    
    for t in range(h_test):
        if t % 20 == 0:
            print(f"  Step {t}/{h_test}")
        
        # Online setting: we have data up to train_size + t
        current_X_train = X[:train_size + t]
        current_Y_train = Y[:train_size + t].flatten()
        current_X_test = X[train_size + t : train_size + t + 1]
        
        for name, model_func in models.items():
            pred = model_func(current_Y_train, current_X_train, 1, current_X_test)
            predictions[name].append(pred[0])
            
    plt.figure(figsize=(14, 8))
    plt.plot(np.arange(h_test), Y_test, color='blue', label='True Test Data', linewidth=2)
    
    colors = ['orange', 'green', 'red', 'purple']
    mses = {}
    
    for i, (name, preds) in enumerate(predictions.items()):
        mse = mean_squared_error(Y_test, preds)
        mses[name] = mse
        print(f"{name} MSE: {mse:.4f}")
        plt.plot(np.arange(h_test), preds, color=colors[i], label=f'{name} (MSE: {mse:.2f})', alpha=0.7)
        
    plt.title('Point Prediction Comparison')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('./imgs/compare_forecasters.png')
    plt.close()
    print("Done! Plot saved to ./imgs/compare_forecasters.png")
