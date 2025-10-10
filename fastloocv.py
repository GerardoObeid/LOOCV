import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class FastLOOCV:
    """
    A class to perform Fast Leave-One-Out Cross-Validation (LOOCV).

    Attributes
    ----------
    data : The dataset used for performing LOOCV. Typically this includes features
           and labels required for model training and evaluation.
    """

    def __init__(self, data):
        """
        Initialize the FastLOOCV object.

        Parameters
        ----------
        data : array-like
            Input dataset to be used for LOOCV.
        """
        self.data = data

    def do_fast_loocv(self, k_values, sample_size=None):
        """
        Perform fast leave-one-out cross-validation for a set of K values.

        Parameters
        ----------
        k_values : array-like
            List or array of integer K values to evaluate (e.g., number of neighbors).
        sample_size : int, optional
            Number of samples to randomly select from the training set.
            If None, all available samples are used.

        Returns
        -------
        score : numpy.ndarray
            Vector containing performance scores for each K in k_values.
        elapsed_time : float
            Execution time (in seconds) for running the procedure.

        This Cross Validation (Fast-LOOCV) approach invloves much less computational cost:
            - training_time = number_of_k
            - evaluation_time = 
            - computation_time = trainig_time + evaluation_time
        """
        start_time = time.time()
        score = np.zeros(len(k_values))

        if sample_size:
            indices = np.random.choice(len(self.data["data"]), size=sample_size, replace=False)
            X = self.data["data"][indices]
            Y = self.data["target"][indices]
        else:
            X = self.data["data"]
            Y = self.data["target"]

        # As derived from the Kanagawa's paper we will just train the model once for every possible value of k
        for index, k in enumerate(k_values):
            
            # Model training once for every k, by using k + 1 NN 
            model = KNeighborsRegressor(n_neighbors = k+1, algorithm = 'kd_tree')
            model.fit(X, Y)

            # Model evaluation for each K without retraining the model n times, n being the number of samples
            # for i in range(len(X)):
            #     actual_val = Y[i]
            #     actual_x = X[i]
            #     y_pred = model.predict(actual_x.reshape(1, -1))
            #     score[index] += ((( (k+1) / k) ** 2) * ((actual_val-y_pred)**2))/len(X)
            predictions = model.predict(X)
            score[index] = ((( (k+1) / k) ** 2) * np.sum((predictions -Y)**2))/len(X)


        elapsed_time = time.time() - start_time
        return score, elapsed_time
        
    def do_normal_loocv(self, k_values, sample_size=None):
        """
        Perform standard leave-one-out cross-validation for a set of K values.

        Parameters
        ----------
        k_values : array-like
            List or array of integer K values to evaluate (e.g., number of neighbors).
        sample_size : int, optional
            Number of samples to randomly select from the training set.
            If None, all available samples are used.

        Returns
        -------
        score : numpy.ndarray
            Vector containing performance scores for each K in k_values.
        elapsed_time : float
            Execution time (in seconds) for running the procedure.
        
        This Cross Validation (LOOCV) approach invloves high computational cost:
            - training_time = number_of_k * number of samples (n)
            - evaluation_time = 
            - computation_time = trainig_time + evaluation_time
        """
        start_time = time.time()


        score = np.zeros(len(k_values))
        if sample_size:
            indices = np.random.choice(len(self.data["data"]), size=sample_size, replace=False)
            X = self.data["data"][indices]
            Y = self.data["target"][indices]
        else:
            X = self.data["data"]
            Y = self.data["target"]

        # Iterate in every possible k value
        for index, k in enumerate(k_values):
            # Iteration for every xl, l = 1,...,n 
            for i in range(len(X)):
                actual_val = Y[i]
                actual_x = X[i]
                # Training model for every k and every xl, l = 1,...,n 
                model = KNeighborsRegressor(n_neighbors = k, algorithm = 'kd_tree')
                model.fit(np.delete(X, i, axis=0), np.delete(Y, i, axis=0))

                y_pred = model.predict(actual_x.reshape(1, -1))
                score[index] += ((actual_val - y_pred) ** 2)/len(X)




        elapsed_time = time.time() - start_time
        return score, elapsed_time
