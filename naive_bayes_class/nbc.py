import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class myMixedNB:
    def __init__(self, L=1):
        """
        # Important:
        The mixed class assumes that in the training and prediction set, the categorical features are at the last columns, for simplicity.
        First the continuous features and then the categorical features. In myMixedNB.train, the caller has to input the
        categorical_feats list, which includes the indices of the categorical features.
        In myMixedNB.predict, the parameters for the categorical features are looked for in a list that starts with index 0,
        which corresponds to the first categorical feature. The formula used is
        theta[feature - self.categorical_feats[0]][feat_value][cls]
        to be used as an index, so if the position of the first categorical feature is 15, then we will look for its
        probabilities in params_x_given_y [15-15] = param_x_given_y [0] which is what we wanted.
        Same logic holds for the means and variances for the continuous features but since these are the first ones that appear, the indices hold.
        """
        self.means = None
        self.variances = None
        self.classes = None
        self.params_x_given_y = None
        self.params_y = None
        self.predictions = None

        self.D_categorical = None
        self.categorical_feats = None

        self.L = L
    
    def train(self, X,Y, categorical_feats: list = None, D_categorical=None):
        """
        inputs
        -------
        - categorical_feats: list of the indices of the columns that are categorical in X. Must  be included.
        - D_categorical:  1xM vector. Each element D(m) contains the number of possible values for each column of the ones in
        categorical_feats. Defaults to None, so the algorithm automatically determines it.
        """

        '''if D_categorical is None:
                raise ValueError('D_categorical must be provided for categorical data')'''

        self.categorical_feats = categorical_feats
        n_samples, n_features = X.shape
        categoricals = X[:,categorical_feats].astype(int)


        continuous_feats = [i for i in range(n_features) if i not in categorical_feats]  # indices of continuous features
        continuous = X[:,continuous_feats]

        if self.D_categorical is None: 
            #self.D_categorical = [len(np.unique(categoricals[:, idx])) for idx in range(categoricals.shape[1])]
            self.D_categorical = [2 for _ in range(len(self.categorical_feats))]

        self.train_categorical_part(categorical_data=categoricals, Y=Y)
        self.train_continuous_part(continuous_data=continuous, Y=Y)

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
        means = self.means
        variances = self.variances

        for sample in X: #-> iterates over rows
            max_prob = -np.inf
            predicted_class = None

            for cls in classes:
                
                prob = np.log(pi[cls])
                for feature in range(len(sample)):
                    if feature in self.categorical_feats:
                        feat_value = int(sample[feature])
                        # sample[feature]: this cat. feature could be at index K in the dataset, but
                        # since these features are assumed to be at the right side of the dataset array,
                        # we look at the K - N index in the params array.

                        # using log to avoid numerical underflow
                        try:
                            prob += np.log(theta[feature - self.categorical_feats[0]][feat_value][cls])
                        except IndexError:
                            raise IndexError('Index out of range in parameters array. Does your training set contain all possible cases?')
                    else:
                        prob += np.log(1 / np.sqrt(2 * np.pi * variances[cls][feature])) + (-1 / (2 * variances[cls][feature])) * (sample[feature] - means[cls][feature])**2

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = cls
                    
            predictions.append(predicted_class)
        self.predictions = predictions

    def train_categorical_part(self, categorical_data, Y):
        categoricals = categorical_data
        # useful variables
        n_samples, n_features = categoricals.shape

        classes = [0,1]#np.unique(Y) #-> sorted array of class VALUES
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
            for elem in Y:
                if elem == cls:
                    count += 1
            class_counts[cls] = count
            params_y[cls] = (count + self.L) / (n_samples + self.L*n_classes) 
        
        ### learning parameters θ_ijk ###
        for feature in range(n_features): # feature i, use D_categorical[i] to know how many values it takes
            for feature_value in np.unique(categoricals[:,feature]): # -> sorted array of the unique values feature i takes
                                                        # this is used for training, so we don't need to check for the
                                                        # values we dont see in the train set, probability will remain zero

                for cls in classes:
                    count = 0 # numerator = number of { X_i = x_ij AND Y = y_k }
                    
                    for row in range(n_samples): # we count for all samples
                        if categoricals[row, feature] == feature_value and Y[row] == cls:
                            count += 1
                    
                    theta_ijk =  (count + self.L) / (class_counts[cls] + self.L * self.D_categorical[feature])
                    params_x_given_y[feature][feature_value][cls] = theta_ijk

        ## if something isnt seen before, some values in theta will remain zero, leading to errors in the predictions
        # so we need to manually use the default value as if count in the numerator had been found as zero
        for list_idx in range(len(params_x_given_y)):
            for dictionary_idx in range(len(params_x_given_y[list_idx])):
                for key, value in params_x_given_y[list_idx][dictionary_idx].items():
                    if value == 0:
                        params_x_given_y[list_idx][dictionary_idx][key] = 1 / (class_counts[key] + self.L * self.D_categorical[list_idx])  # default value
        # update class fields
        self.params_x_given_y = params_x_given_y
        self.params_y = params_y
        self.classes = classes

    def train_continuous_part(self, continuous_data, Y):
        """"   
        # Inputs
        - `X` : IxM matrix of variables. Rows correspond to the I samples and columns
        to the M variables.
        - `Y` : Ix1 vector. Y is the class variable you want to predict.
        """
        # useful variables
        n_samples, n_features = continuous_data.shape

        classes = [0,1]#np.unique(Y) #-> sorted array of class VALUES
        n_classes = len(classes)

        class_counts : dict = {cls : 0 for cls in classes} #-> dict, key = class , value = how many times it appears in Y
                        # e.g. { class0: 54, class1: 82, ...}
        for cls in classes:
            count = np.sum(Y == cls)
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
                    if Y[row] == cls:
                        sum += continuous_data[row,feature]
                        sum_squares += continuous_data[row,feature]**2

                mean_ik = sum / class_counts[cls]  # = Sum (xi) / n
                means[cls][feature] = mean_ik

                var_ik = sum_squares / class_counts[cls] - mean_ik**2  # = Sum (xi)^2 / n  - mean^2
                variances[cls][feature] = var_ik

        self.means = means
        self.variances = variances


class myCategoricalNB:
    def __init__(self, L=1):
        """
        # Inputs
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

        #print(f'predicting \n {theta}')


        for sample in X: #-> iterates over rows
            max_prob = -np.inf
            predicted_class = None

            for cls in classes:
                
                prob = np.log(pi[cls])
                for feature in range(len(sample)):
                    #print(f'theta at feat {feature} = {theta[feature]}')
                    # using log to avoid numerical underflow
                    try:
                        theta_val = theta[feature][sample[feature]][cls]
                    except IndexError:
                        raise IndexError('Index out of range in parameters array. Does your training set contain all possible cases?')


                    #debug log
                    #if theta_val == 0:  print(f'predicting \n feature {feature}, sample {sample}, class {cls}, theta is zero.')
                    #elif theta_val is None: print(f'feature {feature}, class {cls}, theta is None.')

                    
                    prob += np.log(theta_val)

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = cls
                    
            predictions.append(predicted_class)
        self.predictions = predictions
    
    def train(self, X, Y, D_categorical=None):
        """
        - `X` : IxM matrix of variables. Rows correspond to the I samples and columns
        to the M variables.
        - `Y` : Ix1 vector. Y is the class variable you want to predict.
        - D_categorical : 1xM vector. Each element D(m) contains the number of possible
        different values that the categorical variable m can have. This vector is
        ignored if X dtype = ”continuous”. Default = None
        Does
        -----
        Fills the model fields with the learned parameters
        """
        if D_categorical is None:
                raise ValueError('D_categorical must be provided for categorical data')

        # useful variables
        n_samples, n_features = X.shape

        classes = np.unique(Y) #-> sorted array of class VALUES
        n_classes = len(classes)


        # initialize θ_ijk list of lists of dictionaries
        params_x_given_y = [[ dict.fromkeys(classes, 0) for j in range(D_categorical[i])] for i in range(n_features)]
                # -> theta [ feature : int] [ feature value : int ] [ class : anything]

        #print(f'\n theta initialized: {params_x_given_y}')
        ### learning parameters π_k ###
        # number of { Y = y_k } / number of samples

        params_y = { cls : 0 for cls in classes}
        class_counts = {cls : 0 for cls in classes}

        for cls in classes:
            count = 0
            for elem in Y:
                if elem == cls:
                    count += 1
            class_counts[cls] = count
            params_y[cls] = (count + self.L) / (n_samples + self.L*n_classes) 
        
        ### learning parameters θ_ijk ###
        for feature in range(n_features): # feature i, use D_categorical[i] to know how many values it takes
            for feature_value in np.unique(X[:,feature]): # -> sorted array of the unique values feature i takes
                                                        # this is used for training, so we don't need to check for the
                                                        # values we dont see in the train set, probability will remain zero

                for cls in classes:
                    count = 0 # numerator = number of { X_i = x_ij AND Y = y_k }
                    
                    for row in range(n_samples): # we count for all samples
                        if X[row, feature] == feature_value and Y[row] == cls:
                            count += 1
                    
                    theta_ijk =  (count + self.L) / (class_counts[cls] + self.L * D_categorical[feature])
                    params_x_given_y[feature][feature_value][cls] = theta_ijk

                    #debugging
                    #print(f'learning \n feature {feature_value}, class {cls}, theta {theta_ijk}')

        ## if something isnt seen before, some values in theta will remain zero, leading to errors in the predictions
        # so we need to manually use the default value as if count in the numerator had been found as zero
        for list_idx in range(len(params_x_given_y)):
            for dictionary_idx in range(len(params_x_given_y[list_idx])):
                for key, value in params_x_given_y[list_idx][dictionary_idx].items():
                    if value == 0:
                        params_x_given_y[list_idx][dictionary_idx][key] = 1 / (class_counts[key] + self.L * D_categorical[list_idx])  # default value

        # update class fields
        self.params_x_given_y = params_x_given_y
        self.params_y = params_y
        self.classes = classes

class myGaussianNB:
    def __init__(self):
        """

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
    
    def train(self, X, Y):
        """"   
        # Inputs
        - `X` : IxM matrix of variables. Rows correspond to the I samples and columns
        to the M variables.
        - `Y` : Ix1 vector. Y is the class variable you want to predict.
        """
        # useful variables
        n_samples, n_features = X.shape

        classes = np.unique(Y) #-> sorted array of class VALUES
        n_classes = len(classes)

        class_counts : dict = {cls : 0 for cls in classes} #-> dict, key = class , value = how many times it appears in Y
                        # e.g. { class0: 54, class1: 82, ...}
        for cls in classes:
            count = np.sum(Y == cls)
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
                    if Y[row] == cls:
                        sum += X[row,feature]
                        sum_squares += X[row,feature]**2

                mean_ik = sum / class_counts[cls]  # = Sum (xi) / n
                means[cls][feature] = mean_ik

                var_ik = sum_squares / class_counts[cls] - mean_ik**2  # = Sum (xi)^2 / n  - mean^2
                variances[cls][feature] = var_ik

        self.means = means
        self.variances = variances
        self.params_y = params_y
        self.classes = classes