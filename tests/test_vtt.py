import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from vtt import VTT


def test_vtt():
	""" Test TT """

	N = 20 # number of documents
	F = 15 # number of features
	
	ones = np.ones((N*0.5,F*0.5), dtype=np.int)
	zeros = np.zeros((N*0.5,F*0.5), dtype=np.int)
	NER_counts = np.array(np.random.rand(N,1) * 10, dtype=np.int)
	X1 = np.concatenate((ones,zeros), axis=1)
	X2 = np.concatenate((zeros,ones), axis=1)
	X = np.concatenate((X1,X2),axis=0)
	X = np.concatenate((X, NER_counts), axis=1)

	y1 = [1]*len(ones)
	y2 = [0]*len(zeros)
	y = np.hstack((y1,y2))

	#X = (np.random.randn(N,F) > 0.5).astype('float')

	print 'X'
	print X
	print 'y'
	print y
	print '---'

	X_train = X[:N*0.9,:]
	X_test = X[N*0.9:,:]

	y_train = y[:N*0.9]
	y_test = y[N*0.9:]
	#X_train = X_test = X
	#y_train = y_train = y

	X_test =[[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 10],
 			[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 11],
 			[0, 0, 0, 0, 0, 0 ,0 ,1 ,1, 1, 1, 1, 1, 1, 12],
 			[0, 0, 0, 0, 0, 0, 0, 1, 1 ,1 ,1, 1, 1, 1, 13]]
	
	y_test = [1,1,0,0]

	"""
	X_train = np.array([[1,1,1,0,0,0],
						[1,1,1,0,0,0],
						[0,0,0,1,1,1],
						[0,0,0,1,1,1]])
	y_train = np.array([1,1,-1,-1])
	"""

	X_train = csr_matrix(X_train)
	X_test = csr_matrix(X_test)
	
	classifier = VTT()
	classifier.set_params(b_14=10)
	classifier.fit(X_train, y_train)
	print classifier.get_params()

	"""
	X_test = np.array([[1,1,1,0,0,0],
					   [0,0,0,1,1,1]])
	"""

	print 'X_train'
	print X_train.todense()
	print 'y_train'
	print y_train
	print 'X_test'
	print X_test.todense()
	print 'y_test'
	print y_test
	X_test = csr_matrix(X_test)

	y_predict = classifier.predict(X_test)
	print y_test, y_predict
	accuracy = classifier.score(X_test, y_test)
	print 'Accuracy: ', accuracy

	assert ( y_predict == y_test).all()

