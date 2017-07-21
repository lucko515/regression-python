import numpy as np

class LinearRegressioN(object):
    
    #in constructor - defining all hyperparameter
    def __init__(self):
        self.learning_rate = 0.0001
        #for accurecy play with how many iterations our gradient descent will run
        #100k = same output as in sklearn
        self.no_of_iter = 100000
        self.starting_b = 0
        self.starting_m = 0
        
    def train(self, X, y):
        #y = m*x + b
        self.X_train = X
        self.y_train = y
        
        #calling gradient descent function, and output of it is going to be our the best possible (according to our dataset) M and B
        self.new_b, self.new_m = self.gradient_descent(self.starting_b, self.starting_m, self.X_train, self.y_train, self.learning_rate, self.no_of_iter)
        
        return self.new_b, self.new_m
   
    #main function for gradient descent
    #INPUTS: STARTING b and m (alwyas start with b=0 and m=0)
    #        X and y -  Our training features and labels
    #        learn_rate - is another hyperparam which will define "how fast" are it is going to find good B and M
    #        no_of_iter - how many times our algorith is going to run
    def gradient_descent(self, b, m, X, y, learn_rate, no_of_iter):
        bG = b
        mG = m 
        
        for i in range(no_of_iter):
            bG, mG = self.gradient_descent_step(bG, mG, X, y, learn_rate)
        
        return bG, mG
    #helper function for Gradient descent
    #      current m and b - for each iteration its going to use that B and M
    #      X, y - training features and labels
    #      learn_rate - how fast it is going to move and to find the best possible M and B
    def gradient_descent_step(self, b, m, X, y, learn_rate):
        gradient_b = 0
        gradient_m = 0
        
        for i in range(len(X)):
            gradient_b += -(2/float(len(X)) * (y[i] - ((m*X[i]) + b)))
            gradient_m += (2/float(len(X))) * (-X[i]) * (y[i] - ((m*X[i]) + b))
        
        new_b = b - (learn_rate * gradient_b)
        new_m = m - (learn_rate * gradient_m)
        return new_b, new_m
                            
    #predict function to predict labels (y - values) for TEST set
    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            #y = m*x + b
            y_pred.append(self.new_m*X[i] + self.new_b)

        return y_pred