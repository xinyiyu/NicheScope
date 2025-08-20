import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import cauchy, chi2
from momentchi2 import hbe


def cov_test(counts,infomat,mc_cores=1):
    Xinfomat = infomat - np.mean(infomat, axis=0)
    loc_inv = np.linalg.inv(np.dot(Xinfomat.T, Xinfomat))
    kmat_first = np.dot(Xinfomat, loc_inv)
    LocDim = infomat.shape[1]
    Klam = np.linalg.eig(np.dot(Xinfomat.T, kmat_first))[0]
    if isinstance(Klam[0], complex):
        # print('get complex eig')
        Klam = np.ones(infomat.shape[1])
        
    EHL = np.dot(counts, Xinfomat)
    numCell = Xinfomat.shape[0]

    adjust_nominator = np.sum(counts**2, axis=1)
    
    vec_stat = np.diagonal(np.dot(EHL, np.dot(loc_inv, EHL.T)))* numCell / adjust_nominator


    vec_ybar = np.mean(counts, axis=1)
    vec_ylam = 1 - numCell * vec_ybar**2 / adjust_nominator
    vec_daviesp = np.array(
        [get_pval(i, vec_ylam, Klam, vec_stat) for i in range(counts.shape[0])]
    )
    res_covtest = pd.DataFrame({
        'stat': vec_stat,
        'pval': vec_daviesp
    })
    
    gene_ids = res_covtest[~res_covtest['pval'].isna()].index.tolist()
    res_covtest = res_covtest[~res_covtest['pval'].isna()]

    res_covtest.loc[res_covtest['pval'] < 1e-300, 'pval'] = 1e-300
    res_covtest.loc[res_covtest['pval'] > 1, 'pval'] = 1

    # Extract all statistics and p-values
    allstat = res_covtest['stat'].values
    allpvals = res_covtest['pval'].values
    if len(allpvals.shape) == 1:
        allpvals = allpvals[:, None]

    # Apply the ACAT function to all p-values
    comb_pval = np.array([ACAT(pvals) for pvals in allpvals])

    # Adjust the p-values using the Benjamini-Yekutieli method
    pBY = multipletests(comb_pval, method='fdr_by')[1]

    # Create a DataFrame for combined and adjusted p-values
    joint_pval = pd.DataFrame({
        'combinedPval': comb_pval,
        'adjustedPval': pBY
    })

    # Assuming gene_ids and allstat are defined elsewhere in your code
    res_covtest = {
        'gene_ids': gene_ids,
        'stats': allstat,
        'res_stest': allpvals,
        'res_mtest': joint_pval
    }
    return res_covtest
    
    
def ACAT(Pvals, Weights=None):
    # Check if there are NA values
    if np.any(np.isnan(Pvals)):
        raise ValueError("Cannot have NAs in the p-values!")

    # Check if Pvals are between 0 and 1
    if np.any(Pvals < 0) or np.any(Pvals > 1):
        raise ValueError("P-values must be between 0 and 1!")

    # Check if there are pvals that are either exactly 0 or 1
    is_zero = np.any(Pvals == 0)
    is_one = np.any(Pvals == 1)

    if is_zero and is_one:
        raise ValueError("Cannot have both 0 and 1 p-values!")
    if is_zero:
        return 0.0
    if is_one:
        print("Warning: There are p-values that are exactly 1!")
        return 1.0

    # Default: equal weights. If not, check the validity of the user supplied weights and standardize them.
    if Weights is None:
        Weights = np.ones(len(Pvals)) / len(Pvals)
    elif len(Weights) != len(Pvals):
        raise ValueError("The length of weights should be the same as that of the p-values")
    elif np.any(Weights < 0):
        raise ValueError("All the weights must be positive!")
    else:
        Weights = Weights / np.sum(Weights)

    # Check if there are very small non-zero p values
    is_small = Pvals < 1e-16
    if np.sum(is_small) == 0:
        cct_stat = np.sum(Weights * np.tan((0.5 - Pvals) * np.pi))
    else:
        cct_stat = np.sum((Weights[is_small] / Pvals[is_small]) / np.pi)
        cct_stat += np.sum(Weights[~is_small] * np.tan((0.5 - Pvals[~is_small]) * np.pi))

    # Check if the test statistic is very large
    if cct_stat > 1e+15:
        pval = (1 / cct_stat) / np.pi
    else:
        pval = 1 - cauchy.cdf(cct_stat)

    return pval


def get_pval(igene, lambda_G, lambda_K, allstat):
    # Sort the product of lambda_G[igene] and lambda_K in decreasing order
    Zsort = np.sort(lambda_G[igene] * lambda_K)[::-1]

    try:
        pout = 1 - hbe(coeff=Zsort, x=allstat[igene])
        if pout <= 0:
            pout = calculate_q(allstat[igene], Zsort)

    except Exception as e:
        pout = np.nan  # Return NaN if there's an error

    return pout


def calculate_q(q, lambda_, h=None, delta=None):
    # Set default values for h and delta
    if h is None:
        h = np.ones(len(lambda_))
    if delta is None:
        delta = np.zeros(len(lambda_))

    r = len(lambda_)

    # Check conditions
    if np.any(delta < 0):
        raise ValueError("All non-centrality parameters in 'delta' should be positive!")
    if len(h) != r:
        raise ValueError("lambda and h should have the same length!")
    if len(delta) != r:
        raise ValueError("lambda and delta should have the same length!")

    # Calculate c1, c2, c3, c4
    c1 = np.sum(lambda_ * h) + np.sum(lambda_ * delta)
    c2 = np.sum(lambda_**2 * h) + 2 * np.sum(lambda_**2 * delta)
    c3 = np.sum(lambda_**3 * h) + 3 * np.sum(lambda_**3 * delta)
    c4 = np.sum(lambda_**4 * h) + 4 * np.sum(lambda_**4 * delta)

    s1 = c3 / (c2 ** (3/2))
    s2 = c4 / (c2 ** 2)
    muQ = c1
    sigmaQ = np.sqrt(2 * c2)
    tstar = (q - muQ) / sigmaQ

    if s1**2 > s2:
        a = 1 / (s1 - np.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        l = a**2 - 2 * delta
    else:
        a = 1 / s1
        delta = 0
        l = c2**3 / c3**2

    muX = l + delta
    sigmaX = np.sqrt(2) * a
    # Calculate Qq using the chi-squared distribution
    Qq = chi2.sf(tstar * sigmaX + muX, df=l, loc=delta)

    return Qq

