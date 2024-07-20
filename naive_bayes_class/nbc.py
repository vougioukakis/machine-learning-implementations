import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class myCategoricalNB:
    def __init__(self,X, Y, D_categorical = None, L=1):
        """
        # Inputs
        - `X` : IxM matrix of variables. Rows correspond to the I samples and columns
        to the M variables.
        - `Y` : Ix1 vector. Y is the class variable you want to predict.
        - D_categorical : 1xM vector. Each element D(m) contains the number of possible
        different values that the categorical variable m can have. This vector is
        ignored if X dtype = ”continuous”. Default = None
        - L: the smoothing parameter. Defaults to 1.


        # Fields
        ## Model fields

        - params_x_given_y : list[list[dict]] theta_ijk parameters, theta [ feature : int] [ feature value : int ] [ class : anything ]
        - params_y : pi parameters, (occurences of class) / (number of samples) if L=0
        - classes: the possible class values
        - predictions: 1xJ array of the predicted classes for the corresponding sample in the predict dataset, JxM.
        - probabilities_0  : probabilities P(Xi = 0 | Y = 0) for every feature
        - probabilities_1 : probabilities P(Xi = 0 | Y = 1) for every feature

        ## Input data fields
        - D_categorical :  1xM vector. Each element D(m) contains the number of pos-
        sible different values that the categorical variable m can have. This vector is
        ignored if X dtype = ”continuous”. Default = None
        - X : IxM matrix of variables. Rows correspond to the I samples and columns
        to the M variables.
        - Y:Ix1 vector. Y is the class variable you want to predict.

        ## Hyperparameters
        - L: the smoothing parameter. Defaults to 1.
        """
        self.L = L

        # model fields
        self.params_x_given_y = []
        self.params_y = []
        self.classes = []
        self.predictions = []
        self.probabilities_0 = []
        self.probabilities_1 = []

        # input data fields
        self.D_categorical = D_categorical
        self.X = X
        self.Y = Y

    def predict(self, X):
        """
        Inputs
        -------
        • `X`: JxM matrix of variables. Rows correspond to the J samples and columns
        to the M variables.

        Does
        ------
        Fills the self.predictions field with the prediction results.
        """
        predictions = []
        theta = self.params_x_given_y
        pi = self.params_y
        classes = self.classes


        for sample in X: #-> iterates over rows
            max_prob = -np.inf
            predicted_class = None

            for cls in classes:
                
                prob = np.log(pi[cls])
                for feature in range(len(sample)):
                    # using log to avoid numerical underflow
                    prob += np.log(theta[feature][sample[feature]][cls])

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = cls
                    
            predictions.append(predicted_class)
        self.predictions = predictions
    
    def train(self):
        """
        Does
        -----
        Fills the model fields with the learned parameters
        """
        if self.D_categorical is None:
                raise ValueError('D_categorical must be provided for categorical data')

        # useful variables
        n_samples, n_features = self.X.shape

        classes = np.unique(self.Y) #-> sorted array of class VALUES
        n_classes = len(classes)


        # initialize θ_ijk list of lists of dictionaries
        params_x_given_y = [[ dict.fromkeys(classes, 0) for j in range(self.D_categorical[i])] for i in range(n_features)]
                # -> theta [ feature : int] [ feature value : int ] [ class : anything]

        ### learning parameters π_k ###
        # number of { Y = y_k } / number of samples

        params_y = { cls : 0 for cls in classes}
        class_counts = {cls : 0 for cls in classes}

        for cls in classes:
            count = 0
            for elem in self.Y:
                if elem == cls:
                    count += 1
            class_counts[cls] = count
            params_y[cls] = (count + self.L) / (n_samples + self.L*n_classes) 
        
        ### learning parameters θ_ijk ###
        for feature in range(n_features): # feature i, use D_categorical[i] to know how many values it takes
            for feature_value in np.unique(self.X[:,feature]): # -> sorted array of the unique values feature i takes
                                                        # this is used for training, so we don't need to check for the
                                                        # values we dont see in the train set, probability will remain zero

                for cls in classes:
                    count = 0 # numerator = number of { X_i = x_ij AND Y = y_k }
                    
                    for row in range(n_samples): # we count for all samples
                        if self.X[row, feature] == feature_value and self.Y[row] == cls:
                            count += 1
                    
                    theta_ijk =  (count + self.L) / (class_counts[cls] + self.L * self.D_categorical[feature])
                    params_x_given_y[feature][feature_value][cls] = theta_ijk

        

        ### this part is for question (d) ###

        # probabilities P(Xi = 0 | Y = cls) for every feature
        probabilities_0 = []
        probabilities_1 = []

        for cls in [0, 1]: # fill the probabilities_0, 1 arrays
            for feature in range(n_features):
                feature_values = np.unique(self.X[:, feature])
                y_values = [params_x_given_y[feature][value][cls] for value in feature_values]
                if cls==0: probabilities_0.append(params_x_given_y[feature][0][cls])
                else: probabilities_1.append(params_x_given_y[feature][0][cls])


        # update class fields
        self.params_x_given_y = params_x_given_y
        self.params_y = params_y
        self.classes = classes
        self.probabilities_0 = probabilities_0
        self.probabilities_1 = probabilities_1

class myGaussianNB:
    def __init__(self, X, Y):
        """
        # Inputs
        - `X` : IxM matrix of variables. Rows correspond to the I samples and columns
        to the M variables.
        - `Y` : Ix1 vector. Y is the class variable you want to predict.

        # Fields
        ## Model fields

        - predictions: 1xJ array of the predicted classes for the corresponding sample in the predict dataset, JxM.
        - params_y : pi parameters
        - classes: the possible class values
        - means : dictionary of lists, keys are class names, values are the means for each feature.
        example, means = { 'class0' : [μ feat0, μ feat1,...],  'class1' : [μ feat0,...], ...}
        - variances : similar to means.
        - probabilities_0 = : probabilities P(Xi = 0 | Y = 0) for every feature
        - probabilities_1 = [] : probabilities P(Xi = 0 | Y = 1) for every feature


        """
        self.means = None
        self.variances = None
        self.params_y = None
        self.classes = None
        self.predictions = []

        self.X = X
        self.Y = Y

    def predict(self, X):
        predictions = []
        means = self.means
        variances = self.variances
        params_y = self.params_y
        classes = self.classes

        for sample in X: #-> iterates over rows
            max_prob = -np.inf
            predicted_class = None

            for cls in classes:
                prob = np.log(params_y[cls])
                for feature in range(len(sample)):
                    # using log to avoid numerical underflow
                    # prob here is the sum of the logs of gaussians
                    prob += np.log(1 / np.sqrt(2 * np.pi * variances[cls][feature])) + (-1 / (2 * variances[cls][feature])) * (sample[feature] - means[cls][feature])**2

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = cls
                    
            predictions.append(predicted_class)

        self.predictions = predictions
    
    def train(self):
        # useful variables
        n_samples, n_features = self.X.shape

        classes = np.unique(self.Y) #-> sorted array of class VALUES
        n_classes = len(classes)

        class_counts : dict = {cls : 0 for cls in classes} #-> dict, key = class , value = how many times it appears in Y
                        # e.g. { class0: 54, class1: 82, ...}
        for cls in classes:
            count = np.sum(self.Y == cls)
            class_counts[cls] = count
        
        # initialize μ_ik and variances as a dictionary of lists like this:
        # means = { 'class0' : [μ feat0, μ feat1,...], 
        #           'class1' : [μ feat0,...],
        #          ...}
        # all means and variances for all features are initialized as zero

        means = {cls : [0 for feature in range(n_features)] for cls in classes}
        variances = {cls : [0 for feature in range(n_features)] for cls in classes}

        ### learning parameters π_k ###
        # number of { Y = y_k } / number of samples

        params_y = { cls : 0 for cls in classes}
        for cls in classes:
            params_y[cls] = class_counts[cls] / n_samples
        
        ### learning parameters μ_ik, σ squared ###
        for feature in range(n_features):
            for cls in classes:
                sum = 0 # numerator = sum of rows for which Yk = cls
                sum_squares = 0 # for variance
                
                for row in range(n_samples): # we count for all samples - rows
                    if self.Y[row] == cls:
                        sum += self.X[row,feature]
                        sum_squares += self.X[row,feature]**2

                mean_ik = sum / class_counts[cls]  # = Sum (xi) / n
                means[cls][feature] = mean_ik

                var_ik = sum_squares / class_counts[cls] - mean_ik**2  # = Sum (xi)^2 / n  - mean^2
                variances[cls][feature] = var_ik

        self.means = means
        self.variances = variances
        self.params_y = params_y
        self.classes = classes