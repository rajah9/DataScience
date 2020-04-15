import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

from typing import Callable, List
from Util import Util

Strings = List[str]

"""
Interesting Python features:
* This is the signature for train_model:
** train_model(X_train, y_train, f_classifier: Callable[[], list]=None, seed:int=42) -> list:
** the f_classifier is passed in (or a default is used)
* Subclasses Util and does a super().__init__(). This gets it a self.logger.
"""

# Following are one-time imports that may be commented out after they are downloaded.
# download('stopwords')

class DataScienceUtil(Util):
    def __init__(self):
        super().__init__()
        self.param_grid = None
        self.grid = None
        self._vectorizer = None

    @staticmethod
    def convert_lower_case(sent: str) -> str:
        return np.char.lower(sent)

    @staticmethod
    def remove_stop_words(sent:str) -> str:
        stop_words = stopwords.words('english')
        words = word_tokenize(str(sent))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    @staticmethod
    def train_model(X_train, y_train, f_classifier: Callable[[], list]=None, seed:int=0, param_dict:dict={}) -> list:
        """
        Train a model with the given classifier. Return the predicted y for the test set.
        :param X: matrix of input variables
        :param y: response vector
        :param test_frac: proportion used for testing; 0.2 = 20%
        :param f_classifier: function passed in as a classifier. Will default to LR, but here's SVC:
        from sklearn.svm import SVC
        classifier = DataScienceUtil.train_model(X_train, y_train, f_classifier=SVC, seed=0)
        :param seed:
        :return: predicted output for the test set.
        """
        if seed:
            param_dict['random_state'] = seed
        classifier_func = f_classifier or LogisticRegression

        classifier = classifier_func(**param_dict)
        classifier.fit(X_train, y_train)
        return classifier

    @staticmethod
    def model_predict(classifier, X_test:list) -> list:
        """
        Predict using the given classifier
        :param classifier:
        :param X_test:
        :return:
        """
        y_predict = classifier.predict(X_test)
        return y_predict

    @staticmethod
    def train_test_split(X: list, y: list, test_frac: float = 0.2, seed: int = 42) -> list:
        """
        Train a model with the given classifier. Return the predicted y for the test set.
        :param X: matrix of input variables
        :param y: response vector
        :param test_frac: proportion used for testing; 0.2 = 20%
        :param seed: integer seed
        :return: four lists: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_frac, random_state = seed)

    @staticmethod
    def classification_report(y_test:list, y_predict:list) -> list:
        return classification_report(y_test, y_predict)

    @staticmethod
    def scale(X: list, y: list=None, f_scaler: Callable[[], list]=None,) -> list:
        """
        Scale the list using f_scale. Here is how to use the MinMaxScalar:
          from sklearn.preprocessing import MinMaxScaler
          actual = DataScienceUtil.scale(X=weight, f_scaler=MinMaxScaler())

        :param X:
        :param f_scale:
        :return: transformed X
        """
        scaler_func = f_scaler or StandardScaler()
        param_dict = {'X': X}
        if y:
            param_dict['y': y]
        return scaler_func.fit_transform(**param_dict)

    def build_param_grid(self, C_list:list=[0.1, 1, 10, 100], gamma_list:list=[1, 0.1, 0.01, 0.001], kernel_list:list=['rbf']):
        """
        Return a dictionary of the parameter grid to be searched.
        Notes form udemy unit 35 (about Support Vector Machines).
          * C Parameter is the trade-off between generalizing and overfitting.
            * Small C is loose / soft margin / straight line
            * Large C is strict with a potential to overfit.
            * Might do a grid search from c=1 to c=1000
          * Gamma puts higher weight or influence on the data points close to the boundary.
            * Large gamma => More reach = more points, further away from the boundary, more general
            * Small gamma -> Less reach, right at the boundary, very specific
          * Kernel
            * rbf = Radial Basis Function
            * linear

        :param C_list:
        :param gamma_list:
        :param kernel_list:
        :return:
        """
        ans = {}
        ans['C'] = C_list
        ans['gamma'] = gamma_list
        ans['kernel'] = kernel_list
        self.param_grid = ans
        return ans

    def grid_search_and_fit(self, X: list, y: list, classifier=SVC, is_refit:bool=True, verbose_level:int=4):
        if not self.param_grid:
            self.logger.warning('Uninitialized param_grid. Please call build_param_grid first.')
            return None
        self.grid = GridSearchCV(classifier, self.param_grid, refit=is_refit, verbose=verbose_level)
        ans = self.grid.fit(X=X, y=y)
        self.logger.info(f'best grid parameters are: {self.grid.best_params_} with a score of {self.grid.best_score_}')
        return self.grid

    def grid_predict(self, X: list):
        if not self.grid:
            self.logger.warning('Uninitialized grid. Please call grid_search_and_fit first.')
            return None
        return self.grid.predict(X)

    def label_encoder(self, y: list) -> list:
        """
        Encode a list of labels.
        :param y:
        :return:
        """
        self._le = LabelEncoder()
        y = self._le.fit_transform(y)
        return y

    def label_names(self) -> Strings:
        """
        Return a list of how the labels were encoded.
        :return:
        """

        try:
            if self._le:
                return self._le.classes_.tolist()
        except AttributeError:
            self.logger.warning('AttributeError: LabelEncoder was not found.')
            self.logger.warning('No LabelEncoder. Please call label_encoder first.')
            return None

    @staticmethod
    def count_vector(df:pd.DataFrame, column_name:str, y:list=None):
        """
        Return a count vector for the documents in column_name
        :param df:
        :param column_name:
        :param y:
        :return:
        """
        vectorizer = CountVectorizer()
        # print(vectorizer.get_feature_names())
        ans = vectorizer.fit_transform(raw_documents=df[column_name], y=y)
        return ans

    def count_vectorizer(self, df:pd.DataFrame, column_name:str, y:list=None):
        """
        Return a count vector for the documents in column_name
        :param df:
        :param column_name:
        :param y:
        :return:
        """
        self._vectorizer = CountVectorizer()
        # print(vectorizer.get_feature_names())
        ans = self._vectorizer.fit_transform(raw_documents=df[column_name], y=y)
        return ans

    def vectorizer_features(self) -> list:
        """
        Return a list of the feature names.
        :return:
        """
        if self._vectorizer:
            return self._vectorizer.get_feature_names()
        self.logger.warning('Uninitialized vector. Please call count_vectorizer first.')