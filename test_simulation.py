import numpy as np
import UQTools.generate as gen

if __name__ == "__main__":
    print("Testing autoregressive covariate generation...")
    X = gen.generate_X(100, 1, distribution='normal', ar_coeffs=[0.7, -0.2])
    print(f"  X shape: {X.shape}")

    print("Testing autoregressive target generation...")
    func = lambda x: np.sin(2*np.pi*x.mean(axis=1)) + 2*np.cos(3*np.pi*x.mean(axis=1))
    Y = gen.generate_y(X, func, noise_distribution='normal', ar_coeffs=[0.5])
    print(f"  Y shape: {Y.shape}")
    
    print("Done!")
