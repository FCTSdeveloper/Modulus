import numpy as np

class LinearRegression:
    def __init__(self, model_type = 'leastsquares', learning_rate = 0.01, iters = 1000):
        self.learning_rate = learning_rate
        self.iters = iters
        self.model_type = model_type
        if self.model_type == 'gradientdescent':
            self.model = LinearRegression_gradientdescent(learning_rate = self.learning_rate, iters = self.iters)
        elif self.model_type == 'leastsquares':
            self.model = LinearRegression_Leastsquares()
        else:
            raise ValueError("지원하지 않는 모델 유형입니다.")
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
class LinearRegression_gradientdescent:
    def __init__(self,learning_rate=0.01,iters=1000):
        self.lr=learning_rate
        self.iters=iters
        self.weights=None
        
    def fit(self,X,y):
        n_samples=len(X)
        ones=np.ones(len(X))
        features=np.c_[ones,X]
        self.weights = np.zeros(features.shape[1])
        
        for i in range(self.iters):
            y_predicted=np.dot(features,self.weights.T)
            error=y_predicted-y
            dw = (2 / n_samples) * np.dot(features.T,error)
            self.weights -= self.lr * dw
    
    def predict(self,X):
        ones=np.ones(len(X))
        features=np.c_[ones,X]
        y_predicted=np.dot(features,self.weights.T)
        return y_predicted
    
class LinearRegression_Leastsquares:
    def __init__(self):
        self.weights = None 
        self.bias = None

    def fit(self, X, y):
        X = np.column_stack([np.ones((X.shape[0], 1)), X])
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.bias = self.weights[0]

    def predict(self, X):
        X = np.column_stack([np.ones((X.shape[0], 1)), X])
        y_pred = X.dot(self.weights)
        return y_pred