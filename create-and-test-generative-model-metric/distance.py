import torch
import numpy as np

SCALE = 1000
GAMMA = 1 / (2 * 10**2)

def mmd(gen_mat, real_mat):
    
    gen_mat = torch.from_numpy(gen_mat)
    real_mat = torch.from_numpy(real_mat)
    
    gen_mat_mul = torch.matmul(gen_mat, gen_mat.T)
    real_mat_mul = torch.matmul(real_mat, real_mat.T)
    gen_real_mat_mul = torch.matmul(gen_mat, real_mat.T)

    gen_mat_square_norm = torch.diag(gen_mat_mul)
    real_mat_square_norm = torch.diag(real_mat_mul)
    
    kernel_xx = (torch.unsqueeze(gen_mat_square_norm, 1) + torch.unsqueeze(gen_mat_square_norm, 0) - 2 * gen_mat_mul)
    expectation_kernel_xx = torch.mean(torch.exp(-GAMMA * kernel_xx))
    
    kernel_yy = (torch.unsqueeze(real_mat_square_norm, 1) + torch.unsqueeze(real_mat_square_norm, 0) - 2 * real_mat_mul)
    expectation_kernel_yy = torch.mean(torch.exp(-GAMMA * kernel_yy))
    
    kernel_xy = (torch.unsqueeze(gen_mat_square_norm, 1) + torch.unsqueeze(real_mat_square_norm, 0) - 2 * gen_real_mat_mul)
    expectation_kernel_xy = torch.mean(torch.exp(-GAMMA * kernel_xy))
    
    return SCALE * (expectation_kernel_xx + expectation_kernel_yy - 2 * expectation_kernel_xy)


def fd(x, y):
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    
    print('x', x.shape)
    print('y', y.shape)
    
    mu1 = np.mean(x, axis=0)
    sigma1 = np.cov(x, rowvar=False)
    mu2 = np.mean(y, axis=0)
    sigma2 = np.cov(y, rowvar=False)
    
    mu1 = torch.from_numpy(mu1)
    sigma1 = torch.from_numpy(sigma1)
    mu2 = torch.from_numpy(mu2)
    sigma2 = torch.from_numpy(sigma2)
    
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c