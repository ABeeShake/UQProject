

def block_bootstrap(X,Y,B=1000, block_size = 10):
    """
    X: t x d
    Y: t x 1
    B: number of bootstrap samples
    block_size: length of blocks to sample

    Resamples a time series {(x_s,y_s)}_{s=1}^t using block bootstrap

    * Setting 1: Resample blocks of pairs (x_s,y_s) with replacement
    * Setting 2: Resample blocks of x_s and y_s independently with replacement
    """

    return None


def find_block_length(X,Y):
    """
    X: t x d
    Y: t x 1

    Find optimal block length for block bootstrap
    * Idea 1: minimize MSE between original and bootstrapped series
    * Idea 2: use autocorrelation cutoff
    """
    