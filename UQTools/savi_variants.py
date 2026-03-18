import numpy as np
from scipy.stats import gaussian_kde, norm, t
from UQTools.savi import BootstrapSAVI

class BootstrapSAVI_Window(BootstrapSAVI):
    def __init__(self, alpha=0.1, window=50):
        super().__init__(alpha)
        self.window = window

    def _get_predictive_samples(self, bootstrap_forecasts):
        mean_forecast = np.mean(bootstrap_forecasts)
        res = self.past_residuals[-self.window:] if len(self.past_residuals) > 0 else []
        if len(res) < 5:
            spread = bootstrap_forecasts - mean_forecast
            samples = mean_forecast + spread * 2.0
            return samples
        else:
            return mean_forecast + np.array(res)

class BootstrapSAVI_KDE(BootstrapSAVI):
    def _p_values(self, predictive_samples, candidates):
        N = len(predictive_samples)
        candidates = np.atleast_1d(candidates)
        if N < 5:
            return super()._p_values(predictive_samples, candidates)
            
        try:
            kde = gaussian_kde(predictive_samples)
            pvals = np.zeros(len(candidates))
            for i, y in enumerate(candidates):
                F = kde.integrate_box_1d(-np.inf, y)
                p = 2 * min(F, 1 - F)
                pvals[i] = max(p, 1.0 / (N + 1))
            return pvals
        except np.linalg.LinAlgError:
            # Fallback if singular covariance
            return super()._p_values(predictive_samples, candidates)

class BootstrapSAVI_Kelly(BootstrapSAVI):
    def __init__(self, alpha=0.1):
        super().__init__(alpha)
        self.past_pvals = []
        
    def _optimize_kappa(self):
        if len(self.past_pvals) == 0:
            return 0.5
        kappas = np.linspace(0.1, 0.9, 9)
        best_k = 0.5
        max_w = -np.inf
        p_arr = np.array(self.past_pvals)
        for k in kappas:
            w = np.sum(np.log(k * (p_arr ** (k - 1))))
            if w > max_w:
                max_w = w
                best_k = k
        return best_k

    def confidence_interval(self, bootstrap_forecasts):
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        min_s = np.min(pred_samples)
        max_s = np.max(pred_samples)
        pad = max(max_s - min_s, 1e-3)
        candidates = np.linspace(min_s - 1.5*pad, max_s + 1.5*pad, 1000)
        
        pvals = self._p_values(pred_samples, candidates)
        kappa = self._optimize_kappa()
        evalues = kappa * (pvals ** (kappa - 1))
        
        mask = (self.capital * evalues) < self.threshold
        if not np.any(mask):
            return None
        valid = candidates[mask]
        return valid.min(), valid.max()

    def update(self, y_true, bootstrap_forecasts):
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        pvals = self._p_values(pred_samples, [y_true])
        p = pvals[0]
        
        kappa = self._optimize_kappa()
        evalue = kappa * (p ** (kappa - 1))
        
        self.capital *= evalue
        self.capital = max(min(self.capital, 1e10), 1e-10)
        
        self.past_pvals.append(p)
        mean_forecast = np.mean(bootstrap_forecasts)
        self.past_residuals.append(y_true - mean_forecast)

class BootstrapSAVI_EnbPIHybrid(BootstrapSAVI):
    def __init__(self, alpha=0.1):
        super().__init__(alpha)
        self.past_model_residuals = [] 

    def _get_predictive_samples(self, bootstrap_forecasts):
        if len(self.past_model_residuals) < 5:
            mean_forecast = np.mean(bootstrap_forecasts)
            spread = bootstrap_forecasts - mean_forecast
            return mean_forecast + spread * 2.0
            
        # Add the most recent residual of each individual bootstrap model to its forecast
        recent_res = self.past_model_residuals[-1]
        samples = bootstrap_forecasts + recent_res
        return samples
        
    def update(self, y_true, bootstrap_forecasts):
        self.past_model_residuals.append(y_true - bootstrap_forecasts)
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        pvals = self._p_values(pred_samples, [y_true])
        evalue = 1.0 / (2 * np.sqrt(pvals[0]))
        self.capital = max(min(self.capital * evalue, 1e10), 1e-10)

class BootstrapSAVI_Combined(BootstrapSAVI):
    """
    Combines Adaptive Window, KDE Smoothing, and E-value Mixing to successfully mitigate 
    Test Capital decay and narrow the interval width safely.
    """
    def __init__(self, alpha=0.1, window=50, gamma=0.5):
        super().__init__(alpha)
        self.window = window
        self.gamma = gamma

    def _get_predictive_samples(self, bootstrap_forecasts):
        mean_forecast = np.mean(bootstrap_forecasts)
        res = self.past_residuals[-self.window:] if len(self.past_residuals) > 0 else []
        if len(res) < 5:
            spread = bootstrap_forecasts - mean_forecast
            return mean_forecast + spread * 2.0
        return mean_forecast + np.array(res)

    def _p_values(self, predictive_samples, candidates):
        N = len(predictive_samples)
        candidates = np.atleast_1d(candidates)
        if N < 5:
            return super()._p_values(predictive_samples, candidates)
            
        try:
            kde = gaussian_kde(predictive_samples)
            pvals = np.zeros(len(candidates))
            for i, y in enumerate(candidates):
                F = kde.integrate_box_1d(-np.inf, y)
                p = 2 * min(F, 1 - F)
                pvals[i] = max(p, 1e-6) # Allow smaller p-values for tighter tails
            return pvals
        except np.linalg.LinAlgError:
            return super()._p_values(predictive_samples, candidates)

    def _mixed_evalue(self, p):
        # E-variable mixing to prevent catastrophic log-wealth decay
        e_raw = 1.0 / (2 * np.sqrt(p))
        return self.gamma * e_raw + (1 - self.gamma)

    def confidence_interval(self, bootstrap_forecasts):
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        min_s = np.min(pred_samples)
        max_s = np.max(pred_samples)
        
        # We can search a tighter grid than the 1.5 pad
        pad = max(max_s - min_s, 1e-3)
        candidates = np.linspace(min_s - 1.0*pad, max_s + 1.0*pad, 1000)
        
        pvals = self._p_values(pred_samples, candidates)
        evalues = self._mixed_evalue(pvals)
        
        mask = (self.capital * evalues) < self.threshold
        if not np.any(mask):
            return None
            
        valid = candidates[mask]
        return valid.min(), valid.max()

    def update(self, y_true, bootstrap_forecasts):
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        pvals = self._p_values(pred_samples, [y_true])
        
        evalue = self._mixed_evalue(pvals[0])
        
        self.capital *= evalue
        self.capital = max(min(self.capital, 1e10), 1e-10)
        
        mean_forecast = np.mean(bootstrap_forecasts)
        self.past_residuals.append(y_true - mean_forecast)


class AdaptiveBettingSAVI(BootstrapSAVI):
    """
    Online Newton-Hedged Confidence Sequence (ONH-CS).
    
    This class implements an anytime-valid prediction interval using 
    Adaptive Conformal Inference, KDE smoothing, and ONS wealth optimization.
    """
    def __init__(self, alpha=0.1, window_size=50):
        super().__init__(alpha)
        self.window_size = window_size
        # ONS Optimization Parameters
        self.lambda_t = 0.0  # Initial betting parameter lambda
        self.S_t = 1.0  # Running proxy for the Hessian (sum of squared gradients)
        self.eta = 1.0  # Learning rate for ONS

    def _get_predictive_samples(self, bootstrap_forecasts):
        mean_forecast = np.mean(bootstrap_forecasts)
        res = self.past_residuals[-self.window_size:] if len(self.past_residuals) > 0 else []
        if len(res) < 5:
            spread = bootstrap_forecasts - mean_forecast
            return mean_forecast + spread * 2.0
        return mean_forecast + np.array(res)

    def _p_values(self, predictive_samples, candidates):
        N = len(predictive_samples)
        candidates = np.atleast_1d(candidates)
        if N < 5:
            return super()._p_values(predictive_samples, candidates)
            
        try:
            kde = gaussian_kde(predictive_samples)
            pvals = np.zeros(len(candidates))
            for i, y in enumerate(candidates):
                F = kde.integrate_box_1d(-np.inf, y)
                p = 2 * min(F, 1 - F)
                pvals[i] = max(p, 1e-10) # Safety floor to prevent infinite logs in ONS
            return pvals
        except np.linalg.LinAlgError:
            return super()._p_values(predictive_samples, candidates)

    def get_e_value(self, p_val, lam):
        """
        Linear betting e-value: E = 1 + lambda * (0.5 - p).
        """
        return 1 + lam * (0.5 - p_val)

    def update_params(self, p_val, e_val):
        """
        Online Newton Step (ONS) update for lambda.
        """
        # Gradient of negative log-wealth: - (0.5 - p) / (1 + lam * (0.5 - p))
        grad = -(0.5 - p_val) / e_val
        self.S_t += grad**2
        
        # Newton step update
        self.lambda_t = self.lambda_t - (1/self.eta) * (1/self.S_t) * grad
        
        # Projection onto valid betting range (-2, 2) to keep E_t > 0
        self.lambda_t = np.clip(self.lambda_t, -1.9, 1.9)

    def confidence_interval(self, bootstrap_forecasts):
        """
        Constructs the interval by finding candidate y where 
        capital * E_t(y) < 1/alpha.
        """
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        
        min_s = np.min(pred_samples)
        max_s = np.max(pred_samples)
        pad = max(max_s - min_s, 1e-3)
        candidates = np.linspace(min_s - 1.5*pad, max_s + 1.5*pad, 1000)
        
        p_values = self._p_values(pred_samples, candidates)
        
        # Test which candidates in the grid survive the martingale threshold
        # We use the optimized lambda_t from the previous step.
        e_values = np.array([self.get_e_value(p, self.lambda_t) for p in p_values])
        
        mask = (self.capital * e_values) < self.threshold
        if not np.any(mask):
            return None
            
        accepted_y = candidates[mask]
        return np.min(accepted_y), np.max(accepted_y)

    def update(self, y_true, bootstrap_forecasts):
        # 1. Calculate p-value of the actual observation
        pred_samples = self._get_predictive_samples(bootstrap_forecasts)
        p_true = self._p_values(pred_samples, [y_true])[0]
        
        # 2. Update the actual capital (Wealth)
        e_val = self.get_e_value(p_true, self.lambda_t)
        self.capital *= e_val
        self.capital = max(min(self.capital, 1e10), 1e-10)
        
        # 3. Update lambda for the next timestep (Learning from the error)
        self.update_params(p_true, e_val)
        
        # 4. Manage residuals via parent class past_residuals
        mean_forecast = np.mean(bootstrap_forecasts)
        self.past_residuals.append(y_true - mean_forecast)


class TamedONSSAVI(BootstrapSAVI):
    """
    Optimized SAVI using Tamed ONS betting and Predictive Variance Scaling.
    Replaces KDE with a fast Gaussian approximation for O(1) p-value lookups.
    """
    def __init__(self, alpha=0.1, window_size=50):
        self.alpha = alpha
        self.threshold = 1.0 / alpha
        self.window_size = window_size
        self.past_residuals = []
        self.capital = 1.0 # M_t
        
        # ONS Parameters for Tamed Betting
        self.lam = 0.0
        self.S_t = 1.0
        
    def _get_p_value(self, y_candidate, mean, std):
        """ 
        Fast Gaussian p-value lookup. 
        Using Predictive Variance Scaling to handle heteroskedasticity.
        """
        # Under the null, (y - mean)/std ~ N(0, 1)
        z = (y_candidate - mean) / (1.5*std + 1e-6)
        #EDIT 1: changed norm.cdf to t.cdf
        p = 2 * (1 - t.cdf(abs(z), df=5))
        return np.clip(p, 1e-10, 1.0)

    def _tamed_e_value(self, p, lam):
        """
        Quadratic 'Tamed' E-value: log(E) is lower-bounded by the quadratic.
        This provides more stability than the linear 1 + lambda*(0.5-p).
        """
        x = 0.5 - p
        # E = exp(lambda * x - (lambda**2 / 2) * x**2)
        return np.exp(lam * x - (lam**2 / 2) * (x**2))

    def confidence_interval(self, bootstrap_forecasts):
        mean = np.mean(bootstrap_forecasts)
        # Calculate ensemble standard deviation to scale the search
        std = np.std(bootstrap_forecasts) + 1e-6
        
        # Create a search grid around the mean scaled by current uncertainty
        # This mirrors the 'Predictive Distribution' logic but is faster
        grid = np.linspace(mean - 10*std, mean + 10*std, 500)
        
        pvals = self._get_p_value(grid, mean, std)
        evalues = self._tamed_e_value(pvals, self.lam)
        
        # C_t = { y : M_t * E(y) < 1/alpha }
        mask = (self.capital * evalues) < self.threshold
        
        if not np.any(mask):
            return mean - std, mean + std # Fallback
            
        valid = grid[mask]
        return valid.min(), valid.max()

    def update(self, y_true, bootstrap_forecasts):
        mean = np.mean(bootstrap_forecasts)
        std = np.std(bootstrap_forecasts) + 1e-6
        
        p_true = self._get_p_value(y_true, mean, std)
        x_t = 0.5 - p_true
        
        # 1. Update Capital using the Tamed E-value
        gamma = 0.8
        e_raw = self._tamed_e_value(p_true, self.lam)
        e_val = gamma*e_raw + (1-gamma)
        self.capital *= np.clip(e_val,1e-10,1e10)
        
        # 2. ONS Update for lambda
        # Gradient of log(E_t) w.r.t lambda: x_t - lambda * x_t^2
        grad = x_t - self.lam * (x_t**2)
        self.S_t += grad**2
        
        # Update lambda and clip to prevent divergence
        limit = 0.5
        self.lam = np.clip(self.lam + (grad / self.S_t), -limit, limit)
        
        # 3. Store residual (optional for other logic)
        self.past_residuals.append(y_true - mean)
        if len(self.past_residuals) > self.window_size:
            self.past_residuals.pop(0)