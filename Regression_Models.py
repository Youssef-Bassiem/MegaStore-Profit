import pickle

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class RegressionModels:
    @staticmethod
    def poly(x_train, y_train, x_test, y_test):
        poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

        # fit model
        poly_model.fit(x_train, y_train)
        print("Polynomial Score : ", poly_model.score(x_test, y_test))
        print("Mean Square Error poly", metrics.mean_squared_error(y_test, poly_model.predict(x_test)))

        # saving Polynomial Model
        filename = 'Models/Regression/Polynomial_Regression_Model.sav'
        pickle.dump(poly_model, open(filename, 'wb'))

        # Reading the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        print("pickle result : ", loaded_model.score(x_test, y_test))

    @staticmethod
    def ridge(x_train, y_train, x_test, y_test):
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(x_train, y_train)

        print("Ridge Score : ", ridge_model.score(x_test, y_test))
        print("Mean Square Error ridge", metrics.mean_squared_error(y_test, ridge_model.predict(x_test)))

        # saving Ridge Model
        filename = 'Models/Regression/Ridge_Regression_Model.sav'
        pickle.dump(ridge_model, open(filename, 'wb'))

        # Reading the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        print("pickle result : ", loaded_model.score(x_test, y_test))
