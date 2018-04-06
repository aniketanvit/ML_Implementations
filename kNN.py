import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X, k):
    dists = self.compute_distances(X)
    return self.predict_labels(dists, k)

  def compute_distances(self, X):
    dists = euclidean_distances(X, self.X_train)
    return dists

  def predict_labels(self, dists,k):
    y_predict = np.zeros(dists.shape[0])
    for i in range(0, dists.shape[0]):
      neighbor_labels = np.array(self.y_train[np.argsort(dists[i,:],axis=0)[:k]])
      neighbor_labels[i] = np.bincount(neighbor_labels).argmax()
    return y_predict