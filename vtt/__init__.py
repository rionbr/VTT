# -*- coding: utf-8 -*-
"""
A linear classifier to be used in conjunction with the Scikit Learn python package.

"""
#    Copyright (C) 2016 by
#    Luis Rocha <rocha@indiana.edu>
#    Artemy Kolchinsky <artemyk@gmail.com >
#    Rion Brattig Correia <rionbr@gmail.com>
#    Ian B Wood <ibwood@indiana.edu>
#    All rights reserved.
#    MIT license.
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from scipy.sparse import csr_matrix
import numpy as np

__name__ = 'vtt'
__version__ = '0.3.1'
__release__ = '0.3.1b1'
__authors__ = ' and '.join(['Luis M. Rocha', 'Artemy Kolchinsky', 'Rion Brattig Correia', 'Ian B. Wood'])
__all__ = ['VTT']

class VTT(BaseEstimator, LinearClassifierMixin):
	"""The Variable Trigonometric Threshold (VTT) linear classifier class

	Attributes:
		coef_ (array-like) : Feature weights. Also known as the coefficients.
		intercept (array-like) : This is the classifier bias. For a linear classifier also known as the intercept.
	"""
	def __init__(self, weights=None, bias=None, *args, **kwargs):
		self.coef_ = weights # Weights. Has to be named coef_ so scikit-learn will understand
		self.intercept_ = bias
		self.B = {} #pass keys to set_params of the form 'b_{index}' to treat index as an NER count and B[index] as the weight for the NER
		self.y_predict = None
	
	def __get_vtt_angles(self, pvals, nvals):
		""" Fit the angles to the model

		Args:
			pvals (array-like) : positive values
			nvals (array-like) : negative values

		Returns: normalized coef_ values
		"""
		# https://www.khanacademy.org/math/trigonometry/unit-circle-trig-func/inverse_trig_functions/v/inverse-trig-functions--arctan
		angles = np.arctan2(pvals, nvals)-np.pi/4
		norm = np.maximum(np.minimum(angles, np.pi-angles), -1*np.pi-angles)
		norm = csr_matrix(norm)
		# Remove any weight from the NER features. These will be added later.
		for key, value in self.B.items():
			norm[0, key] = 0.
		return norm

  	def fit(self, X, y):
  		""" Fit the VTT classifier model

  		Args:
  			X (sparse matrix, shape = [n_samples, n_features]) : Training data
  			y (array-like, shape = [n_samples]) : Target values
  		"""
		self.classes_ = unique_labels(y)
		X = csr_matrix(X, dtype=bool)#.tocsr()
		pvals = X[np.array(y==1),:].mean(axis=0)
		nvals = X[np.array(y!=1),:].mean(axis=0)
		
		self.coef_ = self.__get_vtt_angles(pvals, nvals).toarray()
		
		pnvals = (nvals + pvals).T
		
		# this if is necessary in case the bias is being set by GridSearchCV
		if self.intercept_ is None:
			self.intercept_ = -(self.coef_.dot(pnvals)/2.0)[0,0]
		
		for b, val in self.B.items():
			self.coef_[0,b] = 1./val

		return self

	def decision_function(self, X):
		""" Predict confidence scores for samples.
		The confidence score for a sample is the signed distance of that
		sample to the hyperplane.
		
		Args:
			X : {array-like, sparse matrix}, shape = (n_samples, n_features)
				Samples.
		Returns:
			array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
			Confidence scores per (sample, class) combination. In the binary
			case, confidence score for self.classes_[1] where >0 means this
			class would be predicted.

		Note: This is a copy of the method in `sklearn.linear_model.base.LinearClassifierMixin()`,
			changed the `check_array` to convert X to boolean.
		"""
		if not hasattr(self, 'coef_') or self.coef_ is None:
			raise NotFittedError("This %(name)s instance is not fitted "
								 "yet" % {'name': type(self).__name__})

		# Number of NER features. These can be int
		n_ner = len(self.B)

		X_fea = X[ : , :-n_ner]
		X_ner = X[ : , -n_ner:]

		X_fea = check_array(X_fea, dtype=bool, accept_sparse='csr')
		X_ner = check_array(X_ner, dtype=int, ensure_min_features=0, accept_sparse='csr')

		n_features = self.coef_.shape[1]
		if X.shape[1] != n_features:
			raise ValueError("X has %d features per sample; expecting %d" % (X.shape[1], n_features))

		scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
		return scores.ravel() if scores.shape[1] == 1 else scores

	def set_params(self, **params):
		""" Set the parameters of the estimator.

		Args:
			bias (array-like) : bias of the estimator. Also known as the intercept in a linear model.
			weights (array-like) : weights of the features. Also known as coeficients.
			NER biases (array-like) : NER entities infering column position on X and bias value. Ex: `b_4=10, b_5=6`.

		Example:
			>>> cls = VTT()
			>>> cls.set_params(b_4=10, b_5=6, b_6=8)
		"""
		if 'bias' in params.keys():
			self.intercept_ = params['bias']
		if 'weights' in params.keys():
			self.coef_ = params['weights']
		for key in params.keys():
			if 'b_' == key[:2]:
				self.B[int(key[2:])] = params[key]

		return self
	
	def get_params(self, deep=True):
		""" Get parameters for the estimator.

		Args:
			deep (boolean, optional) : If True, will return the parameters for this estimator and contained subobjects that are estimators.

		Returns:
			params : mapping of string to any contained subobjects that are estimators.
		"""
		params = {'weights':self.coef_, 'bias':self.intercept_}
		if deep:
			for key, value in self.B.items():
				params['b_'+str(key)] = value
		
		return params

