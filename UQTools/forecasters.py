import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsforecast.models import AutoARIMA

def create_lags(y, X=None, lags=3):
    """
    Create lagged features for regression.
    """
    T = len(y)
    if X is not None:
        d = X.shape[1] if X.ndim > 1 else 1
        X = X.reshape(T, d)
    
    features = []
    targets = []
    
    for t in range(lags, T):
        # Use previous 'lags' values of y as features
        feat = list(y[t-lags:t])
        if X is not None:
            feat.extend(X[t])
        features.append(feat)
        targets.append(y[t])
        
    return np.array(features), np.array(targets)

def fit_predict_regression(model_type, y_train, X_train=None, h=1, X_test=None, lags=3, **kwargs):
    """
    Fits a regression model and predicts h steps ahead iteratively.
    """
    y_train = y_train.flatten()
    T_train = len(y_train)
    
    features_train, targets_train = create_lags(y_train, X_train, lags=lags)
    
    if model_type == 'ridge':
        model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(**kwargs))])
    elif model_type == 'rf':
        model = RandomForestRegressor(**kwargs)
    elif model_type == 'svr':
        model = SVR(**kwargs)
    elif model_type == 'arima':
        model = AutoARIMA(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    if len(targets_train) > 0:
        if model_type == 'arima':
            model.fit(y=y_train, X=X_train)
        else:
            model.fit(X=features_train, y=targets_train)
    else:
        # Not enough data for lags, return naive forecast
        return np.ones(h) * y_train[-1]
    
    predictions = []
    current_y_history = list(y_train)
    
    if model_type == 'arima':
        pred_kwargs = {'h': h}
        if X_test is not None:
            pred_kwargs['X'] = X_test
        forecasts = model.predict(**pred_kwargs)
        if isinstance(forecasts, dict) and 'mean' in forecasts:
            return forecasts['mean']
        return forecasts
        
    for int_h in range(h):
        # Create features for next step using recent history
        feat = list(current_y_history[-lags:])
        if X_test is not None:
            feat.extend(X_test[int_h])
            
        pred = model.predict([feat])[0]
        predictions.append(pred)
        
        # In a real online setting we would observe the true y,
        # but for h-step ahead forecasting we typically use our own prediction if true not available.
        # Since the simulation currently tests h=1 repeatedly online, this loop will typically just run once.
        current_y_history.append(pred)
        
    return np.array(predictions)
