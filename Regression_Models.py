import pickle

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class RegressionModels:
    @staticmethod
    def poly(x_train, y_train, x_test, y_test):
        poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))

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
        # ridge with bad 8 features
        x_train = x_train[
            ["Ship Mode", "Ship Day", "Ship Year", "Order Year", "Customer ID", "Customer Name", "Order ID",
             "Postal Code"]]

        x_test = x_test[
            ["Ship Mode", "Ship Day", "Ship Year", "Order Year", "Customer ID", "Customer Name", "Order ID",
             "Postal Code"]]
        clc = LinearRegression()
        clc.fit(x_train, y_train)
        print("cls Score : ", clc.score(x_test, y_test))

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
