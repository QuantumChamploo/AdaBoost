# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from mnist import MNIST

mndata = MNIST("./data/")
images, labels = mndata.load_training()

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.3,shuffle=True)

class EnsembleClassifier:
	def __init__(self,learners,depth):
		self.learners = learners
		self.depth = depth
		self.models = []
		self.fit = False


		for i in range(learners):
			self.models.append(DecisionTreeClassifier(max_depth=depth))



	def reg_fit(self):
		
		for model in self.models:
			model.fit(X_train,Y_train)

		self.fit = True
		return 


	def shuf_fit(self):

		for model in self.models:
			X_train_prime, X_test_prime, Y_train_prime, Y_test_prime = train_test_split(images, labels, test_size=0.6,shuffle=True)
			model.fit(X_train_prime,Y_train_prime)

		self.fit = True
		return
			
	def base_acc(self):
		if(self.fit):
			base_accs = [model.score(X_test,Y_test) for model in self.models]
			base = np.array(base_accs)
			return np.mean(base)

		else:
			return -1

	def ensm_guess(self,x_val):
		x_fixed = [x_val]

		guesses = [model.predict(x_fixed)[0] for model in self.models]
		#print(guesses)

		np_guesses = np.array(guesses)
		return np.argmax(np.bincount(np_guesses))

	def ensm_acc(self):
		right = 0
		wrong = 0
		for i in range(len(X_test)):
			if(Y_test[i] == self.ensm_guess(X_test[i])):
				#print(Y_test[i])
				right += 1
			else:
				#print(Y_test[i])
				wrong += 1

		return (right/(right+wrong))


