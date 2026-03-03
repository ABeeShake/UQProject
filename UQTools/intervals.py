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

def wsr_confidence_sequences(y_true, mu_preds, alpha=0.1):
    """
    Construct Waudby-Smith & Ramdas (WSR) variance-adapted Confidence Sequences.
    Instead of betting on the raw residuals, WSR bets dynamically scaled by the 
    empirical variance, adapting the betting fraction lambda.
    
    Parameters:
    y_true (np.ndarray): True observations of shape (T,)
    mu_preds (np.ndarray): Point predictions of shape (T,)
    alpha (float): Desired maximum miscoverage rate.
    
    Returns:
    lower_bound (np.ndarray): Sequence of lower bounds
    upper_bound (np.ndarray): Sequence of upper bounds
    """
    T = len(y_true)
    L_seq = np.zeros(T)
    U_seq = np.zeros(T)
    
    alpha_side = alpha / 2.0
    
    # Grid of candidate means to test for inclusion in the CS
    # Dynamically tracking bounds based on expected range
    
    # We will compute the bounds by keeping track of the wealth processes 
    # for a wide grid of theoretical candidate means `mu_test`.
    # A candidate mu is retained in C_t if wealth(mu) < 1/alpha_side
    
    # To make this computationally feasible for online regression, we track 
    # a grid centered around the current prediction.
    
    num_grid = 1000
    
    # Global tracking of wealth over the grid
    # We'll re-center the grid dynamically, but for simplicity here we assume 
    # y_true is bounded in a known range or we pad around the predictions.
    
    # Empirical variance tracking
    running_var = 1.0 # Initialize with prior
    running_mean_r = 0.0
    
    # History
    history_y = []
    history_mu = []
    
    # In practice, solving WSR explicitly for the roots of the wealth process 
    # is complex without knowing the support bounds a,b exactly. 
    # We use a localized grid search for L_t and U_t at each step.
    
    for t in range(T):
        y_t = y_true[t]
        pred_t = mu_preds[t]
        
        # 1. Update empirical variance of residuals
        if t > 0:
            residual = y_true[t-1] - mu_preds[t-1]
            running_mean_r = ((t-1) * running_mean_r + residual) / t
            running_var = ((t-1) * running_var + (residual - running_mean_r)**2) / t
            running_var = max(running_var, 1e-4)
            
        # 2. Heuristic for betting fraction (e.g. ONS scale)
        # lambda_t ~ 1 / sqrt(var_t * t) or similar empirical tuning
        # Here we use a scaled constant adapting to variance
        lam = 0.5 / np.sqrt(max(running_var, 1e-4) * (t + 1))
        
        # 3. Establish a test grid for 'true_mu' around the prediction
        # The true y is theoretically pred_t + noise.
        margin = 5.0 * np.sqrt(running_var)
        test_mus = np.linspace(pred_t - margin, pred_t + margin, num_grid)
        
        if t == 0:
            # Initialize wealth for each candidate true_mu
            wealths_pos = np.ones(num_grid)
            wealths_neg = np.ones(num_grid)
        else:
            # We must evaluate the wealth for each test_mu over all HISTORY
            # This is O(t) per step, overall O(T^2). Tolerable for T=400.
            wealths_pos = np.ones(num_grid)
            wealths_neg = np.ones(num_grid)
            var_s = 1.0
            mean_r_s = 0.0
            
            for s in range(t):
                # We emulate the betting fraction at step s
                if s > 0:
                    res_s = history_y[s-1] - history_mu[s-1]
                    mean_r_s = ((s-1) * mean_r_s + res_s) / s
                    var_s = ((s-1) * var_s + (res_s - mean_r_s)**2) / s
                    var_s = max(var_s, 1e-4)
                
                lam_s = 0.5 / np.sqrt(var_s * (s + 1))
                
                # Test the candidate mu
                # The bet is placed on (Y_s - test_mu). We cap to avoid negative wealth.
                bet_pos = lam_s * (history_y[s] - test_mus)
                bet_neg = -lam_s * (history_y[s] - test_mus)
                
                # Truncate bets strictly to ensure 1+bet > 0
                evals_pos = np.clip(1 + bet_pos, 1e-10, None)
                evals_neg = np.clip(1 + bet_neg, 1e-10, None)
                
                wealths_pos *= evals_pos
                wealths_neg *= evals_neg
                
        # 4. Filter the grid: Retain mu where wealths < 1/alpha_side
        valid_idx = (wealths_pos < 1.0 / alpha_side) & (wealths_neg < 1.0 / alpha_side)
        
        if np.any(valid_idx):
            L_seq[t] = np.min(test_mus[valid_idx])
            U_seq[t] = np.max(test_mus[valid_idx])
        else:
            # If everything is rejected (poor calibration), fallback to a wide bound
            L_seq[t] = pred_t - margin
            U_seq[t] = pred_t + margin
            
        # 5. Record history
        history_y.append(y_t)
        history_mu.append(pred_t)
        
    return L_seq, U_seq

def find_optimal_lambda(history_y, history_mu, lower_bound, upper_bound, alpha=0.05, num_grid=1000):
    """
    Find optimal betting lambda to maximize log wealth on historical residuals.
    
    Parameters:
    history_y (np.ndarray): Historical true observations
    history_mu (np.ndarray): Historical predicted means
    lower_bound (float): Minimum possible value for lambda
    upper_bound (float): Maximum possible value for lambda
    """
    if len(history_y) == 0:
        return 0.0
        
    history_r = history_y - history_mu
    
    # Grid search for the optimal lambda
    grid = np.linspace(lower_bound, upper_bound, num_grid)
    
    wealths = np.zeros(num_grid)
    for i, lam in enumerate(grid):
        evalues = 1 + lam * history_r
        if np.any(evalues <= 0):
            wealths[i] = -np.inf
        else:
            wealths[i] = np.sum(np.log(evalues))
            
    best_idx = np.argmax(wealths)
    
    if wealths[best_idx] == -np.inf:
        return 0.0 # fallback to no-betting
        
    return grid[best_idx]

def savi_confidence_sequences(y_true, mu_preds_bootstrapped, alpha=0.05):
    """
    Construct SAVI confidence sequences dynamically.
    
    Parameters:
    y_true (np.ndarray): True observations of shape (T,)
    mu_preds_bootstrapped (np.ndarray): Bootstrapped predictions of shape (B, T)
    alpha (float): Desired maximum miscoverage rate (e.g., 0.05 for 95% CS)
    
    Returns:
    lower_bound (np.ndarray): Sequence of lower bounds
    upper_bound (np.ndarray): Sequence of upper bounds
    wealth_upper_seq (np.ndarray): Running wealth logs for upper tracking (for visibility)
    wealth_lower_seq (np.ndarray): Running wealth logs for lower tracking (for visibility)
    """
    T = len(y_true)
    L_seq = np.zeros(T)
    U_seq = np.zeros(T)
    
    # A 2-sided interval distributes the alpha budget to upper and lower bounds.
    alpha_side = alpha / 2.0
    
    # wealth processes starts at 1
    wealth_upper = 1.0
    wealth_lower = 1.0
    
    history_y = []
    history_mu = []
    
    for t in range(T):
        # The mean prediction for time t
        mu_t = np.mean(mu_preds_bootstrapped[:, t])
        
        # Determine the safe range for lambda based on the bootstrap distribution support.
        range_t = np.max(mu_preds_bootstrapped[:, t]) - np.min(mu_preds_bootstrapped[:, t])
        range_t = max(range_t, 1e-4) # Base epsilon padding
        
        # Pad bounds significantly to handle unexpectedly large variations out of sample.
        a_t = np.min(mu_preds_bootstrapped[:, t]) - 3.0 * range_t
        b_t = np.max(mu_preds_bootstrapped[:, t]) + 3.0 * range_t
        
        # Max positive lambda (betting Y > mu) is bounded safely
        if mu_t > a_t:
            lambda_max_bound = 1.0 / (mu_t - a_t)
        else:
            lambda_max_bound = 100.0
            
        if b_t > mu_t:
            lambda_min_bound = -1.0 / (b_t - mu_t)
        else:
            lambda_min_bound = -100.0
            
        # We need to find lambda+ > 0 for U_seq and lambda- < 0 for L_seq based on past residuals
        if t == 0:
            lambda_pos = 0.0
            lambda_neg = 0.0
        else:
            hist_y_arr = np.array(history_y)
            hist_mu_arr = np.array(history_mu)
            lambda_pos = find_optimal_lambda(hist_y_arr, hist_mu_arr, lower_bound=0.0, upper_bound=lambda_max_bound)
            lambda_neg = find_optimal_lambda(hist_y_arr, hist_mu_arr, lower_bound=lambda_min_bound, upper_bound=0.0)
            
        
        # Upper Bound derivation
        if lambda_pos > 1e-8:
            U_t = mu_t + (1.0 / (alpha_side * wealth_upper) - 1.0) / lambda_pos
        else:
            U_t = np.inf
            
        # Lower Bound derivation
        if lambda_neg < -1e-8:
            L_t = mu_t + (1.0 / (alpha_side * wealth_lower) - 1.0) / lambda_neg
        else:
            L_t = -np.inf
            
        L_seq[t] = L_t
        U_seq[t] = U_t
        
        # Observation and Supermartingale Update
        y_t = y_true[t]
        
        wealth_upper = wealth_upper * (1 + lambda_pos * (y_t - mu_t))
        wealth_lower = wealth_lower * (1 + lambda_neg * (y_t - mu_t))
        
        wealth_upper = max(wealth_upper, 1e-10)
        wealth_lower = max(wealth_lower, 1e-10)
        
        history_y.append(y_t)
        history_mu.append(mu_t)
        
    return L_seq, U_seq

