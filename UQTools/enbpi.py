import numpy as np

def enbpi_confidence_sequences(y_true, mu_preds_bootstrapped, alpha=0.1):
    """
    Construct Ensemble Batch Prediction Intervals (EnbPI) 
    using Conformal Prediction.
    
    Parameters:
    y_true (np.ndarray): True observations of shape (T,)
    mu_preds_bootstrapped (np.ndarray): Bootstrapped predictions of shape (B, T)
    alpha (float): Desired maximum miscoverage rate.
    
    Returns:
    lower_bound (np.ndarray): Sequence of lower bounds
    upper_bound (np.ndarray): Sequence of upper bounds
    """
    T = len(y_true)
    B = mu_preds_bootstrapped.shape[0]
    
    L_seq = np.zeros(T)
    U_seq = np.zeros(T)
    
    # Aggregated point forecast
    mu_preds = np.mean(mu_preds_bootstrapped, axis=0)
    
    # EnbPI tracks the empirical distribution of absolute residuals.
    # We maintain a sliding window or full history of non-conformity scores.
    # Non-conformity score: e_t = |Y_t - \hat{Y}_t|
    
    residuals_history = []
    
    for t in range(T):
        pred_t = mu_preds[t]
        
        if len(residuals_history) == 0:
            # First step fallback, we don't have residuals yet.
            # Use max variance over bootstrap as a proxy
            proxy_margin = np.max(mu_preds_bootstrapped[:, t]) - np.min(mu_preds_bootstrapped[:, t])
            L_seq[t] = pred_t - max(proxy_margin, 1.0)
            U_seq[t] = pred_t + max(proxy_margin, 1.0)
        else:
            # Compute empirical quantile of past residuals
            # To achieve 1-alpha coverage, we take the ceil((n+1)(1-alpha))/n quantile
            n = len(residuals_history)
            q_level = min(1.0, (n + 1.0) * (1 - alpha) / n)
            
            # Use beta empirical quantile
            q_err = np.quantile(residuals_history, q_level, method='higher')
            
            L_seq[t] = pred_t - q_err
            U_seq[t] = pred_t + q_err
            
        # Update residuals for the NEXT step using the true observation
        y_t = y_true[t]
        residuals_history.append(np.abs(y_t - pred_t))
        
    return L_seq, U_seq
