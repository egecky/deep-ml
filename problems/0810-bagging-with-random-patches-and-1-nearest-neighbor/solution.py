import numpy as np

def random_patches_bagging(X_train, y_train, X_test, sample_indices_list, feature_indices_list):
    """
    Bagging ensemble using Random Patches and 1-NN base learners.
    Returns a list of averaged predictions over the bags.
    """
    summ = 0
    b = 0

    while b < len(sample_indices_list):
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        
        data = X_train[sample_indices_list[b]][:,feature_indices_list[b]] # n_samples, n_features

        query = X_test[:, feature_indices_list[b]] # n_test, n_features

        dists = ((query[:, None, :] - data[None, :, :])**2).sum(axis=-1)

        nn = np.argmin(dists, axis=-1)

        chosen = y_train[sample_indices_list[b]][nn]

        summ += chosen
        b += 1
    
    return summ / b


