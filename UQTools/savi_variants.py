import numpy as np
from scipy.stats import gaussian_kde
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
