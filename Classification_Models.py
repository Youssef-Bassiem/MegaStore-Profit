import pickle

from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


class ClassificationModels:
    @staticmethod
    def adaboost(x_train, y_train, x_test, y_test):
        adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=14), algorithm="SAMME.R",
                                      n_estimators=100)
        adaboost.fit(x_train, y_train)
        print("AdaBoost Score : ", adaboost.score(x_test, y_test))
        print("Mean Square Error AdaBoost", metrics.mean_squared_error(y_test, adaboost.predict(x_test)))

        # saving AdaBoost Model
        filename = 'Models/Classification/AdaBoost_Classification_Model.sav'
        pickle.dump(adaboost, open(filename, 'wb'))

        # Reading the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        print("pickle result : ", loaded_model.score(x_test, y_test))

    @staticmethod
    def tree(x_train, y_train, x_test, y_test):
        tre = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10)
        tre.fit(x_train, y_train)
        print("tree Score : ", tre.score(x_test, y_test))
        print("Mean Square Error tree", metrics.mean_squared_error(y_test, tre.predict(x_test)))

        # saving Tree Model
        filename = 'Models/Classification/Tree_Classification_Model.sav'
        pickle.dump(tre, open(filename, 'wb'))

        # Reading the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        print("pickle result : ", loaded_model.score(x_test, y_test))

    @staticmethod
    def naive(x_train, y_train, x_test, y_test):
        naive = MultinomialNB(alpha=0.00001, force_alpha=True, fit_prior=True)
        naive.fit(x_train, y_train)

        print("Naive Score : ", naive.score(x_test, y_test))
        print("Mean Square Error Naive", metrics.mean_squared_error(y_test, naive.predict(x_test)))

        # saving Tree Model
        filename = 'Models/Classification/Naive_Classification_Model.sav'
        pickle.dump(naive, open(filename, 'wb'))

        # Reading the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        print("pickle result : ", loaded_model.score(x_test, y_test))
