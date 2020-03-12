from unittest import TestCase
from DataScienceUtil import DataScienceUtil
import pandas as pd
import numpy as np
from statistics import mean, pstdev
from random import randrange
import logging
from LogitUtil import logit
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from keras.datasets import cifar10
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from nltk import download
from sklearn.datasets import load_boston, make_regression
from Util import Util
from re import match

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
* In test_scale, calculates a z-score.
** Note that it uses the population standard deviation (pstdev)
** See comment on https://stackoverflow.com/a/21505523/509840 
* In test_scale, reshapes an array.
"""

class TestDataScienceUtil(TestCase):
    def my_test_set(self, n:int=10):
        def response(x:int) -> float:
            return x * 2 + 3
        x = [randrange(50,100) for i in range(n)]
        y = [[response(w)] for w in x]
        X = [[el] for el in x]
        return np.array(X), np.array(y).reshape(n,) # X is a numpy matrix and y is a numpy vector.

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        plt.subplot(894)
        plt.imshow(x_train[10], cmap=plt.get_cmap('gray'))
        plt.subplot(234)
        plt.imshow(x_train[11], cmap=plt.get_cmap('gray'))
        plt.subplot(654)
        plt.imshow(x_train[12], cmap=plt.get_cmap('gray'))
        plt.subplot(751)
        plt.imshow(x_train[13], cmap=plt.get_cmap('gray'))
        plt.show()

    def load_housing(self):
        bostondata = load_boston()
        return bostondata

    def load_diabetes(self):
        data = pd.read_csv(r"C:\temp\diabetes.csv")
        # Modify the data to add labels
        label = data['Outcome']
        features = data.drop('Outcome', 1)
        # split into input (x) and output (y) vars
        dsu = DataScienceUtil()

        x_train, x_test, y_train, y_test = dsu.train_test_split(X=data, y=label, test_frac = 0.2, seed=42)
        print (data.shape)
        # Create the model (a basic sequential).
        model = Sequential()
        model.add(Dense(12, input_dim=9, activation = 'relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Now fit the model
        model.fit(x_train, y_train, epochs=20, batch_size=10)

    def keras_solution(self):
        boston_data = load_boston()
        boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
        boston['AGE'] = boston_data.target
        # prep a dataset
        np.random.seed(7)
        data = pd.read_csv("C:\temp\diabetes.csv")
        # Modify the data to add labels
        label = data['Outcome']
        features = data.drop('Outcome', 1)
        # split into input (x) and output (y) vars
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state=42)
        # create and compile the model
        # Create the model (a basic sequential).
        model = Sequential()
        model.add(Dense(12, input_dim=9, activation = 'relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Now fit the model
        model.fit(x_train, y_train, epochs=20, batch_size=10)
        # see the scores.
        score = model.evaluate(x_test, y_test, verbose=0)
        print (f'Test loss: {score[0]}. Test accuracy: {score[1]}')
        # Find the min / max scalar values
        scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
        # Make a prediction
        xpred, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
        xpred = scalarX.transform(xpred)
        ypred = model.predict(xpred)
        for i, x in enumerate(xpred):
            print (f'X={x}. Predicted:{ypred[i]}')

    def test_load_data(self):
        self.skipTest("not needed")
        self.load_data()

    def test_load_housing(self):
        # self.skipTest("not needed")
        boston_data = self.load_housing()
        print(boston_data.DESCR)
        df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
        print(df.head())

    def test_convert_lower_case(self):
        test = "For Whom the Bell Tolls"
        expected = "for whom the bell tolls"
        self.assertEqual(expected, DataScienceUtil.convert_lower_case(test))

    def test_remove_stop_words(self):
        test = "for whom the bell tolls"
        expected = "bell tolls"
        self.assertEqual(expected, DataScienceUtil.remove_stop_words(test).strip())

    def test_train_model(self):
        size = 100
        x, y = self.my_test_set(n=size)
        logger.debug(f'len of x and y are {len(x)} and {len(y)}')
        proportion = 0.25
        X_train, X_test, y_train, y_test = DataScienceUtil.train_test_split(x, y, test_frac=proportion)
        self.assertEqual((size, ), y.shape, 'y shape incorrect')
        classifier = DataScienceUtil.train_model(X_train=X_train, y_train=y_train, f_classifier=SVC)
        self.assertEqual(X_train.shape, classifier.shape_fit_)

    def test_model_predict(self):
        self.fail() # TODO

    @logit()
    def test_train_test_split(self):
        size = 100
        x, y = self.my_test_set(n=size)
        logger.debug(f'len of x and y are {len(x)} and {len(y)}')
        proportion = 0.33
        X_train, X_test, y_train, y_test = DataScienceUtil.train_test_split(x, y, test_frac=proportion)
        self.assertEqual((size, 1), x.shape, msg='x shape must agree')
        self.assertEqual((size, ), y.shape, msg='y must be a vector')
        self.assertAlmostEqual(len(x) * proportion, len(X_test), delta=0.7)
        self.assertAlmostEqual(len(y) * (1 - proportion), len(y_train), delta=0.7)
        self.assertEqual(len(X_train) + len(X_test), len(x))

    def test_classification_report(self):
        y_test = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1, 1, 1]
        actual = DataScienceUtil.classification_report(y_test, y_pred)
        pattern = ".*0\s*1.00\s*0.75\s*0.86" # looking for 0  1.00  0.75  0.86
        s = match(pattern, actual)
        self.assertIsNotNone(s)

    def test_scale(self):
        weights = [45, 88, 56, 15, 71]
        weight = np.array(weights).reshape(-1,1)
        def to_z_scores(X: list) -> list:
            df = pd.DataFrame({'w': X})
            x_bar = mean(X)
            std = pstdev(df.w) # Have to use the Population Std. Dev.
            z_score = [(x - x_bar) / std for x in X]
            return z_score
        expected = to_z_scores(X=weights)
        actual = DataScienceUtil.scale(X=weight)

        self.assertListEqual(expected, actual.reshape(len(weights), ).tolist())
        # Test 2. MinMaxScalar
        def to_linear_scale(X: list) -> list:
            list_min = min(X)
            list_max = max(X)
            range = list_max - list_min

            ans = [(1.0 * x - list_min) / range for x in X]
            return ans
        expected = to_linear_scale(X=weight)
        logger.debug(f'here is my local Min/Max: {expected}')
        actual = DataScienceUtil.scale(X=weight, f_scaler=MinMaxScaler())
        for ex, ac in zip(expected, actual.tolist()):
            # ex is an ndarray. Compare the first element to the actual.
            self.assertAlmostEqual(ex[0], ac[0], delta=0.000001)

    def test_label_encoder(self):
        dsu = DataScienceUtil()
        xyzzy = list("xyzzy") # [x y z z y] => [0 1 2 2 1]
        expected = [0, 1, 2, 2, 1]
        actual = dsu.label_encoder(xyzzy)
        self.assertListEqual(expected, list(actual))

    @logit()
    def test_label_names(self):
        dsu = DataScienceUtil()
        # Test 0. Uninitialized.
        expected_log_message = "call label_encoder"
        with self.assertLogs(Util.__name__, level='DEBUG') as cm: # Must use Util.__name__ instead of DataScienceUtil
            ans = dsu.label_names()
            self.assertIsNone(ans)
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
        # Test 1. Normal.
        xyzzy = list("xyzzy") # [x y z z y] => [0 1 2 2 1]
        expected = ['x', 'y', 'z']
        _ = dsu.label_encoder(xyzzy)
        actual = dsu.label_names()
        self.assertListEqual(expected, actual)
