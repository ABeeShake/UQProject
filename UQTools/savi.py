import numpy as np

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


class BootstrapSAVI:
    """
    Maintains SAVI test-supermartingale for sequential forecasting
    using empirical predictive distribution and conformal p-values.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.threshold = 1 / alpha
        self.capital = 1.0
        self.past_residuals = []

    def _get_predictive_samples(self, bootstrap_forecasts):
        mean_forecast = np.mean(bootstrap_forecasts)
        if len(self.past_residuals) < 5:
            # Fallback if not enough history
            spread = bootstrap_forecasts - mean_forecast
            samples = mean_forecast + spread * 2.0
            return samples
        else:
            return mean_forecast + np.array(self.past_residuals)

    def _p_values(self, predictive_samples, candidates):
        N = len(predictive_samples)
        sorted_samples = np.sort(predictive_samples)
        
        candidates = np.atleast_1d(candidates)
        pvals = np.zeros(len(candidates))
        
        for i, y in enumerate(candidates):
            F = np.searchsorted(sorted_samples, y, side="right") / float(N)
            p = 2 * min(F, 1 - F)
            pvals[i] = max(p, 1.0 / (N + 1))
            
        return pvals

    def confidence_interval(self, bootstrap_forecasts):
        """
        Return current SAVI confidence interval for the next observation.
        """
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        
        min_s = np.min(pred_samples)
        max_s = np.max(pred_samples)
        pad = max(max_s - min_s, 1e-3)
        candidates = np.linspace(min_s - 1.5*pad, max_s + 1.5*pad, 1000)
        
        pvals = self._p_values(pred_samples, candidates)
        evalues = 1.0 / (2 * np.sqrt(pvals))
        
        mask = (self.capital * evalues) < self.threshold
        
        if not np.any(mask):
            return None
            
        valid = candidates[mask]
        return valid.min(), valid.max()

    def update(self, y_true, bootstrap_forecasts):
        """
        Update the test-supermartingale using the true observation.
        """
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        pvals = self._p_values(pred_samples, [y_true])
        
        evalue = 1.0 / (2 * np.sqrt(pvals[0]))
        self.capital *= evalue
        self.capital = max(min(self.capital, 1e10), 1e-10) # Prevent overflow/underflow
        
        mean_forecast = np.mean(bootstrap_forecasts)
        self.past_residuals.append(y_true - mean_forecast)