"""To evaluate the performance of any image assessment algorithms, experts recommand to take into account three aspects: 
    - prediction accuracy 
    - prediction monotonicity
    - prediction consistency

To measure prediction accuravy, the reference metrics are the Root Mean Square Error (RMSE) and the Mean Absolute Error (MAE). 
For prediction monotonicity, the Spearman Rank Order Correlation Coefficient (SROCC) and the Kendall Rank Order Correlation Coefficient can be used (KROCC).
Finally for consistency evaluation, the Pearson Linear Correlation Coeffient (PLCC) is the best metric to compute. 

This python script enable to compute all the previously quoted metrics between any IQA algorithm scores and subjective human assessment score which remains the most reliable ground truth score.
"""

# -------------------- PARSING ARGUMENTS -------------------------------------------------------------

import os

from scipy.stats import spearmanr, kendalltau, pearsonr

# -------------------- SUB ASSESSMENT FUNCTIONS ------------------------------------------------------

def SROCC(algo_scores, ref_scores):
    """This function return the SROCC score between two data series

    Args:
        algo_scores (_type_): scores given by an algo
        ref_scores (_type_): reference scores usually subjective humain assessment
    """
    assert len(algo_scores) == len(ref_scores), 'input data series must have same lengths'
    coef, p = spearmanr(algo_scores, ref_scores)
    return(coef)

def KROCC(algo_scores, ref_scores):
    """This function return the KROCC score between two data series

    Args:
        algo_scores (_type_): scores given by an algo
        ref_scores (_type_): reference scores usually subjective humain assessment
    """
    assert len(algo_scores) == len(ref_scores), 'input data series must have same lengths'
    coef, p = kendalltau(algo_scores, ref_scores)
    return(coef)

def PLCC(algo_scores, ref_scores):
    """This function return the PLCC score between two data series

    Args:
        algo_scores (_type_): scores given by an algo
        ref_scores (_type_): reference scores usually subjective humain assessment
    """
    assert len(algo_scores) == len(ref_scores), 'input data series must have same lengths'
    r, p = pearsonr(algo_scores, ref_scores)
    return(r)

def RMSE(algo_scores, ref_scores):
    """This function return the RMSE score between two data series

    Args:
        algo_scores (_type_): scores given by an algo
        ref_scores (_type_): reference scores usually subjective humain assessment
    """
    assert len(algo_scores) == len(ref_scores), 'input data series must have same lengths'
    rmse = (sum((algo_scores - ref_scores)**2)/len(algo_scores))**.5
    return rmse

def MAE(algo_scores, ref_scores):
    """This function return the RMSE score between two data series

    Args:
        algo_scores (_type_): scores given by an algo
        ref_scores (_type_): reference scores usually subjective humain assessment
    """
    assert len(algo_scores) == len(ref_scores), 'input data series must have same lengths'
    mae = sum(abs(algo_scores - ref_scores))/len(algo_scores)
    return(mae)

