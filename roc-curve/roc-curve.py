import numpy as np

def roc_curve(y_true, y_score):

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Sort indices by descending score
    idx = np.argsort(y_score)[::-1]

    # Reorder scores and labels according to sorted indices
    y_score_sorted = y_score[idx]
    y_true_sorted = y_true[idx]

    # Total number of positives and negatives
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)

    # Cumulative true positives and false positives
    TP = np.cumsum(y_true_sorted == 1)
    FP = np.cumsum(y_true_sorted == 0)

    # Compute TPR and FPR
    TPR = TP / P
    FPR = FP / N

    # Find indices where score values change (unique thresholds)
    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    
    # Include the last index to capture the final threshold
    threshold_idxs = np.r_[distinct_indices, len(y_score_sorted) - 1]
    
    # Select TPR, FPR, and thresholds at these indices
    tpr = TPR[threshold_idxs]
    fpr = FPR[threshold_idxs]
    thresholds = y_score_sorted[threshold_idxs]
    
    # Add the starting point (0,0) with threshold = infinity
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    
    return fpr, tpr, thresholds