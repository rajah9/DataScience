import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import tree

from typing import Callable, List
from Util import Util

Strings = List[str]

"""
Interesting Python features:
* This is the signature for train_model:
** train_model(X_train, y_train, f_classifier: Callable[[], list]=None, seed:int=42) -> list:
** the f_classifier is passed in (or a default is used)
* Subclasses Util and does a super().__init__(). This gets it a self.logger.
Rough outline
0. Import Libraries
  a. from DataScienceUtil import DataScienceUtil
  b. dsu = DataScienceUtil()
1. Import Dataset (see PandasUtil)
2. Visualize Dataset (see PlotUtil)
3. Fit the data
  a. for count vectorizer
    1. vec = dsu.count_vectorizer(df, 'text')
    2. logger.debug (f'vector is type: {type(vec)} and shape {vec.shape}')
    3. features = dsu.vectorizer_features()
    4. vec.shape
    5. features[5500:5520]
  b. for Naive bayes
    1. norm_amount = DataScienceUtil.scale(X=df['Amount'].values.reshape(-1,1))
    2. df['Amount_Norm']= norm_amount
4. Training the model
  a - f (See PandasUtil)
  g. X_train, X_test, y_train, y_test = DataScienceUtil.train_test_split(X=X, y=y, test_frac=0.3)
  h. from sklearn.naive_bayes import MultinomialNB
  i. bayes_classifier = DataScienceUtil.train_model(X_train, y_train, f_classifier = MultinomialNB)
  j. y_predict_train = DataScienceUtil.model_predict(classifier=bayes_classifier, X_test=X_test)
5. Evaluating the model
  a. cm = DataScienceUtil.confusion_matrix(y_test, y_predict_train)
  b. plu.heatmap(cm)
  c. report = DataScienceUtil.classification_report(y_test, y_predict_train)
  d. print (f'Classification report:\n{report}')
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
        Example:
          from sklearn.ensemble import RandomForestClassifier
          rfc_dict = {'n_estimators': 150}
          random_forest_classifier = DataScienceUtil.train_model(X_train, y_train, f_classifier = RandomForestClassifier, param_dict=rfc_dict) #RandomForestClassifier(n_estimators=150)

        :param X_train: matrix of input variables
        :param y_train: response vector
        :param f_classifier: function passed in as a classifier. Will default to LR, but here's SVC:
          from sklearn.svm import SVC
          classifier = DataScienceUtil.train_model(X_train, y_train, f_classifier=SVC, seed=0)
        :param seed: integer random seed (defaults to 0)
        :param param_dict: dictionary of any params needed for f_classifier. Random forest takes n_estimators: {'n_estimators': 150}
        :return:
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
        Predict using the given classifier.
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
        Example:
          target_col = 'Kyphosis'
          df_X = pu.drop_col(df, columns=target_col, is_in_place = False) # X drops the target column
          df_y = pu.drop_col_keeping(df, cols_to_keep=target_col, is_in_place=False) # y is only the target vector
          X = pu.convert_dataframe_to_matrix(df_X) # converts to numpy ndarray
          y = pu.convert_dataframe_to_vector(df_y) # reshapes to vector
          X_train, X_test, y_train, y_test = DataScienceUtil.train_test_split(X=X, y=y, test_frac=0.3)
        :param X: matrix of input variables
        :param y: response vector
        :param test_frac: proportion used for testing; 0.2 = 20%
        :param seed: integer seed
        :return: four lists: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_frac, random_state = seed)

    @staticmethod
    def classification_report(y_test:list, y_predict:list) -> str:
        """
        Return a string of the precision, recall, and F1 scores.
        Example usage:
          report = DataScienceUtil.classification_report(y_test, y_predict_train)
          print (f'Classification report:\n{report}')
        :param y_test:
        :param y_predict:
        :return: newline-delimited string
        """
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
        Encode a list of labels. To switch df['Kyphosis'] from "active"/"inactive" to 1/0, try this code:
            dsu = DataScienceUtil()
            df['Kyphosis'] = dsu.label_encoder(df['Kyphosis'])

        :param y: a df vector
        :return:  the vector encoded
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

    def feature_importance(self, decision_tree_classifier: tree, df_X: pd.DataFrame) -> pd.DataFrame:
        """
        For a decision tree, return a dataframe of the feature importances.
        Calling example:
          from sklearn.tree import DecisionTreeClassifier
          decision_tree = DataScienceUtil.train_model(X_train, y_train, f_classifier = DecisionTreeClassifier)
          feature_importance = dsu.feature_importance(decision_tree_classifier=decision_tree_classifier, df_X=df_X)
        :param decision_tree_classifier:
        :return:
        """
        cols = df_X.columns
        feature_importances = pd.DataFrame(decision_tree_classifier.feature_importances_,
                                           index=cols,
                                           columns=['importance']).sort_values('importance', ascending=False)
        return feature_importances

    @staticmethod
    def confusion_matrix(actual: list, predicted: list) -> list:
        """
        This is usually used for y vs. y-predicted.
        Calling example (using the output of this routine as input to a plot)
          y_predict_train = DataScienceUtil.model_predict(classifier=random_forest_classifier, X_test=X_test)
          cm = DataScienceUtil.confusion_matrix(y_test, y_predict_train)
          plu.heatmap(cm)
        :param actual:
        :param predicted:
        :return:
        """
        return confusion_matrix(actual, predicted)