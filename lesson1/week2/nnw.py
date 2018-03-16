import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#%matplotlib inline
def getOrigData():
	#get data
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
	#Reshape to one dimension
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
	#Show info
	print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
	print ("train_set_y shape: " + str(train_set_y.shape))
	print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
	print ("test_set_y shape: " + str(test_set_y.shape))
	print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
	#Standardize
	train_set_x = train_set_x_flatten/255.
	test_set_x = test_set_x_flatten/255.

	return train_set_x,train_set_y,test_set_x,test_set_y,classes

def showOneImage(index,image_set_x,image_set_y,classes):
	plt.imshow(image_set_x[:,index].reshape((64, 64, 3)))
	print("y = " + str(image_set_y[:, index]) + ", it's a '" + classes[np.squeeze(image_set_y[:, index])].decode("utf-8") +  "' picture.")
	plt.show()

def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s

def initializeParams(dim):
	#dim = numLength,1=outputNum
	w = np.zeros((dim,1))
	b = 0
	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))
	return w,b

def doPropagation(w,b,X,Y):
	m = X.shape[1]
	z = np.dot(w.T,X)+b
	A = sigmoid(z)
	cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
	dw = 1 / m * np.dot(X, (A - Y).T)
	db = 1 / m * np.sum(A - Y)
	cost = np.squeeze(cost)
	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	assert(cost.shape == ())

	grids = {
	'dw':dw,
	'db':db
	}
	return grids,cost

def learnParams(w,b,X,Y,learn_times,learning_rate,show_flag=True):
	costs = []
	for i in range(learn_times):
		grids,cost = doPropagation(w,b,X,Y)

		w = w - learning_rate*grids['dw']
		b = b - learning_rate*grids['db']

		if i%100 == 0:
			costs.append(cost)
			if show_flag:
				print("Cost after iteration %i: %f" %(i, cost) )
	params ={'w':w,'b':b}

	return params,grids,costs

def predict(w,b,X):
	m = X.shape[1]
	y_predict = np.zeros((1,m))
	A = sigmoid(np.dot(w.T,X)+b)
	y_predict[A>0.5] = 1
	y_predict[A<=0.5] = 0
	assert(y_predict.shape == (1, m))
	return y_predict

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
	#init
	dim = X_train.shape[0]
	w,b = initializeParams(dim)

	params,grids,costs = learnParams(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

	w = params['w']
	b = params['b']
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d = {"costs": costs,
		"Y_prediction_test": Y_prediction_test, 
		"Y_prediction_train" : Y_prediction_train, 
		"w" : w, 
		"b" : b,
		"learning_rate" : learning_rate,
		"learn_times": num_iterations}

	if print_cost:
		costs = np.squeeze(d['costs'])
		plt.plot(costs)
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(d["learning_rate"]))
		plt.show()
	return d

train_set_x,train_set_y,test_set_x,test_set_y,classes = getOrigData()
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)