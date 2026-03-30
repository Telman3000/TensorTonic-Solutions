def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    
    # Get unique classes
    classes = list(set(y_train))
    n_samples = len(X_train)
    n_features = len(X_train[0])
    
    # Store parameters
    priors = {}
    means = {}
    variances = {}
    
    eps = 1e-9  # to avoid division by zero
    
    # Group data by class
    for c in classes:
        X_c = [X_train[i] for i in range(n_samples) if y_train[i] == c]
        n_c = len(X_c)
        
        # Prior
        priors[c] = n_c / n_samples
        
        # Mean
        mean = [0.0] * n_features
        for x in X_c:
            for j in range(n_features):
                mean[j] += x[j]
        mean = [m / n_c for m in mean]
        means[c] = mean
        
        # Variance (population)
        var = [0.0] * n_features
        for x in X_c:
            for j in range(n_features):
                var[j] += (x[j] - mean[j]) ** 2
        var = [(v / n_c) + eps for v in var]
        variances[c] = var
    
    import math
    
    predictions = []
    
    # Predict
    for x in X_test:
        best_class = None
        best_log_prob = float('-inf')
        
        for c in classes:
            log_prob = math.log(priors[c])
            
            mean = means[c]
            var = variances[c]
            
            # Compute log likelihood
            for j in range(n_features):
                log_prob += -0.5 * math.log(2 * math.pi * var[j])
                log_prob += -((x[j] - mean[j]) ** 2) / (2 * var[j])
            
            # Update best class
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c
        
        predictions.append(best_class)
    
    return predictions