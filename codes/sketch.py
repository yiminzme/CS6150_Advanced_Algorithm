# -*- coding: utf-8 -*-
"""Sketch Based Approximate Nearest Neighbor"""

"""
This class is based on sklearn framework.

Created on Sat Dec  1 19:25:27 2018

@author: sbenw
"""
import numpy as np
from functools import partial

from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.neighbors.base import NeighborsBase, KNeighborsMixin, UnsupervisedMixin
from sklearn.utils._joblib import effective_n_jobs

class SketchKNN(NeighborsBase, KNeighborsMixin, UnsupervisedMixin):
    """ Sketch Based Approximate Nearest Neighbor.
    
    Based on sklearn framework. Read more in the :
    [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors].
    
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
        
    Examples
    --------
    In the following example, we construct a NeighborsClassifier
    class from an array representing our data set and ask who's
    the closest point to [1,1,1]
    
    >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    >>> from sketch import SketchKNN
    >>> neigh = SketchKNN(n_neighbors=1)
    >>> neigh.fit(samples)
    >>> print(neigh.kneighbors([[1., 1., 1.]], return_distance=True))
    (array([[0.5]]), array([[2]]))
    
    As you can see, it returns [[0.5]], and [[2]], which means that the
    element is at distance 0.5 and is the third element of samples
    (indexes start at 0). You can also query for multiple points:
    
    >>> X = [[0., 1., 0.], [1., 0., 1.]]
    >>> neigh.kneighbors(X)
    array([[1],
           [2]]...)
    
    Notes
    -----
    The sketch algorithm is from the paper *Asymmetric Distance Estimation 
    with Sketches for Similarity Search in High-Dimensional Spaces* by 
    Wei Dong, Moses Charikar, and Kai Li.
    """
    def __init__(self, n_neighbors=5, sketch_method='symetric', sketch_size=20, strip_window = 100, random_state=None):
        # NeighborsBase
        super(SketchKNN, self).__init__(n_neighbors=n_neighbors)        
        self.sketch_size = sketch_size
        self.strip_window = strip_window
        self.sketch_method = sketch_method
        self.random_state = random_state
    
    def fit(self, X):
        """Fit the model using X as data
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data, from where find the query point's neighbors.
        """
        super(SketchKNN, self).fit(X)
        self._sketch()
        return self
        
    def kneighbors(self, X, n_neighbors=None, filter_scale=20, return_distance=False):
        """Fast finds the approximate K-neighbors of each point using sketch.
        Returns indices of and distances to the neighbors of each point.
        
        Parameters
        ----------
        X : array-like, shape (n_query, n_features).
            The query point or points.
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        filter_scale : int
            Scale up n_neighbors as number of candidate when filtering using 
            sketch. (defaluts to 20).
        return_distance : boolean, optional. Defaults to False.
            If False, distances will not be returned
        
        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        check_is_fitted(self, ["_fit_X"])
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        X = check_array(X)
        
        n_data = self._fit_X.shape[0]
        n_queries, _ = X.shape
        
        # TODO: Filter candidates using sketch.
        
        # Call sklearn.metrics.pairwise.pairwise_distances_chunked to find
        # results from the filter candidates.
        # TODO: Change self._fit_X to the candidates.
        n_jobs = effective_n_jobs(self.n_jobs)
        reduce_func = partial(self._kneighbors_reduce_func,
                                  n_neighbors=n_neighbors,
                                  return_distance=return_distance)
        # for efficiency, use squared euclidean distances
        kwds = ({'squared': True})
        result = list(pairwise_distances_chunked(
                X, self._fit_X, reduce_func=reduce_func,
                metric=self.effective_metric_, n_jobs=n_jobs,
                **kwds))
        
        if return_distance:
            dist, neigh_ind = zip(*result)
            result = np.vstack(dist), np.vstack(neigh_ind)
        else:
            result = np.vstack(result)
        
        return result
    
    # TODO: Work on this function
    def _sketch(self):
        # self._fit_X
        pass

if __name__ == '__main__':
    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    neigh = SketchKNN(n_neighbors=1)
    neigh.fit(samples)
    print(neigh.kneighbors([[1., 1., 1.]], return_distance=True))