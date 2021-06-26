from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class BuildModel:
    '''
    classifier -> selected classifier
    parameters -> parameters related to classifiers
    X -> working data
    y -> target data
    model -> ML classifier model
    accuracy -> accuracy of the model
    '''
    def __init__(self, classifier, parameters):
        self.classifier = classifier
        self.parameters = parameters

    
    def get_dataset(self, dataset):
        ''' Get the data-set from sklearn dataset.'''
    
        if dataset == "Iris-dataset":
            data = datasets.load_iris()
        elif dataset == "Breast-cancer-dataset":
            data = datasets.load_breast_cancer()
        else:
            data = datasets.load_wine()
    
        self.X = data.data
        self.Y = data.target
        
    
    def get_classifier_model(self):
        '''return the machine-learning model accoring to the selected classifiers'''
        model = None
        if self.classifier == "KNN":
            model = KNeighborsClassifier(n_neighbors = self.parameters["K"])
        elif self.classifier == "SVM":
            model = SVC(C = self.parameters["C"])
        else:
            model = RandomForestClassifier(
                n_estimators = self.parameters["n_estimators"], 
                max_depth = self.parameters["max_depth"],
                random_state = 1243
                )
        self.model = model
        
    
    def train_model(self):
        # classification
        self.get_classifier_model()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=1243)
        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_predict)


    



