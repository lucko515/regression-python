import numpy as np

class LinearRegressionN(object):

	def __init__(self, number_of_interations= 100000, learning_rate= 0.01):
		self.number_of_interations = number_of_interations
		self.learning_rate = learning_rate

	def train(self, X, y):
		''' 
		This function is used to find the best parameters for our M values and B values in linear regression

		Input: X -  Feature matrix with size NxM (N = number of samples, M = number of features for each sample)
			   y -  Values for each sample that we are trying to find (Example: Price of house(s))	
		'''

		#Step 1. Define parameters for Linear Regression formula
		# y = m*X + b <-- Simple Linear Regression
		# y = m1*X1 + m2*X2 + m3*X3 + ... + mn*Xn + b <-- Multiple Linear Regression

		#Get shape[1] == number of features of our train set X and get vector of zeros for our M parameter.
		#In this way we can handle simple linear regresion and Multiple Linear Regression
		self.m = np.array(np.zeros(X.shape[1]))
		self.b = 0

		#Training loop
		for i in range(self.number_of_interations):

			gradient_m = np.zeros(X.shape[1])
			gradient_b = 0

			b = self.b
			m = self.m

			#for each feature
			# for j in range(X.shape[1]):
			gradient_m = np.sum((2/X.shape[0]) * (-X) * (y - (m*X + b)))
			gradient_b = ((2/X.shape[0]) * (-(y - (m*X + b))))
			
			b = b - (gradient_b * self.learning_rate)
			m = m - (gradient_m * self.learning_rate)
			
		
		self.m = m - (gradient_m * self.learning_rate)
		self.b = b - (gradient_b * self.learning_rate)



	def predict(self, X):
		predicted = []
		for i in range(len(X)):
			predicted.append(np.sum(self.m * X[i] + self.b))

		return predicted



# To test Linear Regression uncomment this part
# np.random.seed(0)
# X = np.random.randn(100, 1)
# y = np.random.randn(100, 1)

# li = LinearRegressionN()
# li.train(X, y)

# # kurcina = LinearRegressioN()
# # kurcina.train(X, y)
# import matplotlib.pyplot as plt

# plt.scatter(X, y)
# plt.plot(X, li.predict(X))
# # plt.plot(X, kurcina.predict(X))
# plt.show()

# m = np.ones(X.shape[1])
# b = 0
# # print(m*X + b)