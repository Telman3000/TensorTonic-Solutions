import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    n_train, d = X_train.shape
    n_test = X_test.shape[0]
    
    log_probs = np.zeros((n_test, n_classes))
    
    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]
        
        # log prior
        log_prior = np.log(n_c / n_train)
        
        # Laplace smoothing
        theta = (np.sum(X_c, axis=0) + 1) / (n_c + 2)
        
        log_theta = np.log(theta)
        log_1_minus_theta = np.log(1 - theta)
        
        log_likelihood = (
            X_test * log_theta +
            (1 - X_test) * log_1_minus_theta
        ).sum(axis=1)
        
        log_probs[:, idx] = log_prior + log_likelihood
    
    return log_probs