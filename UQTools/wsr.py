import numpy as np

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
