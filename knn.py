import numpy as np

class Knn():
    def __init__(self, num_neighbors, p=2):
        self.num_neighbors = num_neighbors
        self.p = p
    

    def fit(self, X, Y ):
        """
            Fitting stage is just the process of storing the data
            self.X: Inputs
            self.Y: Targets
        """
        self.X = np.array(X)
        self.Y = np.array(Y)

    def NN(self, x_test):
        """
            Gets the k nearest neighbors
        """
        distances = self.distance(x_test)
        return distances[:self.num_neighbors]

    def distance(self, x):
        distances = [(index, self.minkownski(x, x_train))for  index, x_train in enumerate(self.X)]
        distances.sort(key=lambda x:x[1])
        return distances

    def minkownski(self, x_test, x_train):
        return np.power(np.sum(np.abs(np.array(x_test) - np.array(x_train)) ** self.p), 1 / self.p)

    def predict(self,  X_test):
        """
        """
        predictions = list()
        for x in X_test:
            neighbors = self.NN(x)
            indices = [nn[0] for nn in neighbors]
            predictions.append(sum(self.Y[indices])/self.num_neighbors)


        return predictions

if __name__ == '__main__':
    model = Knn(num_neighbors=4)
    x=[[x,x] for x in range(10)]
    y =[y**2 for y in range(10)]
    model.fit(x,y)

    model.predict([2,2])