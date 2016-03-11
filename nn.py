# -*- coding: utf-8 -*-
import loss_functions
import numpy as np

class TwoLayerNeuralNet(object):

	def __init__(self, input_size, output_size, hidden_size=5, std=1e-4):
		self.b1 = np.zeros(hidden_size)
		self.b2 = np.zeros(output_size)
		self.W1 = std*np.random.randn(input_size,hidden_size)
		self.W2 = std*np.random.randn(hidden_size,output_size)
		self.active_func = lambda x:1.0/(1.0+np.exp(-x))

	def forward_pass(self, X, y):
		z1 = self.W1.T.dot(X) + self.b1
		a1 = self.active_func(z1)
		z2 = self.W2.T.dot(a1) + self.b2
		a2 = z2
		#loss = self.loss_func(a2, y, margin)
		a1 = a1.reshape((a1.shape[0],1))
		a2 = a2.reshape((a2.shape[0],1))

		summary = 0.0
		for i in xrange(a1.shape[0]):
			if i == y:
				continue
			else:
				summary += np.exp(a1[i])

		loss = -a2[y] + np.log(summary)
		# print "z1",z1
		return loss, a1, a2

	def backward_pass(self, X, y, hidden_scores, scores):
		loss = 0.0
		dW1 = np.zeros(self.W1.shape)
		dW2 = np.zeros(self.W2.shape)
		db1 = np.zeros(self.b1.shape)
		db2 = np.zeros(self.b2.shape)

		summary = 0.0
		for i in xrange(hidden_scores.shape[0]):
			if i == y:
				continue
			else:
				summary += np.exp(hidden_scores[i])

		summary2 = 0.0
		for i in xrange(dW2.shape[1]):
			if i == y:
				dW2[:,i] = -hidden_scores
				db2[i] = -1
			else:
				# print dW2[:,i].shape
				dW2[:,i] = (1.0/summary*np.exp(scores[i])*hidden_scores).reshape((hidden_scores.shape[0],))
				db2[i] = 1.0/summary*np.exp(scores[i])
				summary2 += np.exp(scores[i])*self.W2[:,i]

		dLda1 = -self.W2[:,y] + 1.0/summary*summary2

		for i in xrange(dW1.shape[1]):
			dW1[:,i] = dLda1[i]*(1-hidden_scores[i])*hidden_scores[i]*X
			db1[i] = dLda1[i]*(1-hidden_scores[i])*hidden_scores[i]	

		#print db1,db2
		return dW1,dW2,db1,db2

	def train(self, X, y, full_epoch=False, epoch=1000, learning_rate=0.001, lamb=0.001, verbose=False):
		if full_epoch:
			N = X.shape[0]
		else:
			N = epoch
		loss_history = []
		for i in xrange(N):
			# forward pass
			loss, hidden_scores, scores = self.forward_pass(X[i],y[i])
			# loss_history.append(loss)
			# print i,'th train hidden_scores',hidden_scores
			# check
			if verbose:
				loss_sum = 0
				if i % 100 == 0:
					for j in xrange(int(0.7*N),N):
						loss, _, _ = self.forward_pass(X[j],y[j])
						loss_sum += loss
					print i/100+1,loss_sum
					loss_history.append(loss_sum)

			# backward pass
			dW1,dW2,db1,db2 = self.backward_pass(X[i],y[i],hidden_scores,scores)
			print dW2, db2
			# print i,'th train deriative',dW1,dW2,db1,db2
			# update
			self.W1 -= (learning_rate*dW1 + lamb/2*self.W1)
			self.W2 -= (learning_rate*dW2 + lamb/2*self.W2)
			self.b1 -= learning_rate*db1
			self.b2 -= learning_rate*db2
		loss_history = np.array(loss_history)
		#print "W1,",self.W1
		#print "W2,",self.W2
		#print "b1,",self.b1
		#print "b2,",self.b2
		return loss_history, self.W1, self.W2, self.b1, self.b2

	def predict(self, X):
		z1 = self.W1.T.dot(X) + self.b1
		a1 = self.active_func(z1)
		z2 = self.W2.T.dot(a1) + self.b2
		a2 = z2

		return a2.argmax()

	def loss(self, hidden_scores, scores, y):
		summary = 0.0
		loss = 0.0
		for i in xrange(hidden_scores.shape[0]):
			if i == y:
				continue
			else:
				summary += np.exp(hidden_scores[i])
		# if summary < 0.0001:
		# 	summary = 0.01
		loss = -scores[y] + np.log(summary)
		return loss

class Layer(object):

	def __init__(self, size):
		self.input = np.zeros(size)
		self.output = np.zeros(size)
		self.size = size

class NeuralNet(object):

	def __init__(self, structure, std=1e-4):

		# Network parameters
		self.layer = []
		self.W = {}
		self.b = {}
		self.input_size = 0
		self.output_size = 0
		self.active_func = self.sigmoid
		self.dactive_func = self.dsigmoid
		# values
		self.dW = {}
		self.db = {}
		self.delta = {}

		self.sizes = self.translate_structure(structure, std)

	def translate_structure(self, structure, std):

		structure_list = structure.split(',')
		sizes = [int(structure_list[i]) for i in xrange(len(structure_list))]
		# 总层数 输入和输出层也算
		self.layer_num = len(sizes)
		# 申请每一层的空间
		for i in xrange(self.layer_num):
			l = Layer(sizes[i])
			self.layer.append(l)  
		# configure the net
		self.input_size = sizes[0]
		self.output_size = sizes[-1]
		self.output = np.zeros(self.output_size)
		for i in xrange(self.layer_num):
			if i < self.layer_num-1:
				self.W[i] = np.random.randn(sizes[i], sizes[i+1]) * std
				self.b[i] = np.zeros(sizes[i+1])
				self.dW[i] = np.zeros(self.W[i].shape)
				self.db[i] = np.zeros(self.b[i].shape)
				if i > 0:
					self.delta[i] = np.zeros(self.W[i].shape)
		# print self.W,self.b
		# print self.input_size,self.output_size
		return sizes

	def ReLU(self, x):

		x_copy = np.zeros(x.shape)
		for i in xrange(len(x)):
			if x[i] > 0:
				x_copy[i] = x[i]
		return x_copy

	def dReLU(self, x):

		x_copy = np.zeros(x.shape)
		for i in xrange(x.shape[0]):
			if x[i] > 0:
				x_copy[i] = 1
		return x_copy

	def sigmoid(self, x):

		return 1.0/(1 + np.exp(-x))

	def dsigmoid(self, x):

		return (1 - x)*x

	def forward_pass(self, x):

		# 输入层
		self.layer[0].input = x
		self.layer[0].output = x
		# 隐藏层 输出层
		for i in xrange(self.layer_num-2):
			self.layer[i+1].input = self.W[i].T.dot(self.layer[i].output) + self.b[i]
			self.layer[i+1].output = self.active_func(self.layer[i+1].input)
		
		self.layer[self.layer_num-1].input = self.W[self.layer_num-2].T.dot(self.layer[self.layer_num-2].output) + self.b[self.layer_num-2]
		self.layer[self.layer_num-1].output = self.layer[self.layer_num-1].input

		return self.layer[self.layer_num-1].output

	def compute_error(self, output, y):

		# 计算期望输出
		real_output,_ = self.compute_loss(output, y)

		# 输出层残差
		self.delta[self.layer_num-1] = -(real_output - output)

		# 隐藏层残差
		for i in range(self.layer_num-1)[::-1]:
			if i > 0 :
				self.delta[i] = self.W[i].dot(self.delta[i+1])*self.dactive_func(self.layer[i].output)

		return self.delta

	def compute_loss(self, output, y):

		# 构造期望输出
		if self.active_func == self.sigmoid:	# sigmoid 期望输出 n:-0 p:1
			negative = 0
			positive = 1
		elif self.active_func == self.ReLU:		# ReLU 期望输出 n:0 p:x
			negative = 0
			positive = output[y]

		real_output = np.array([negative]*output.shape[0]) # 期望输出
		real_output[y] = positive

		loss = 0.5*(real_output - output)**2

		return real_output, loss

	def predict(self, x):

		output = self.forward_pass(x)
		return output.argmax()

	def backward_pass(self, x, output, y):

		# compute loss
		_, loss = self.compute_loss(output, y)
		# dLdW dLdb dLdout
		self.delta = self.compute_error(output, y)
		# compute the gradients
		for i in xrange(self.layer_num-1):
			a = self.layer[i].output.reshape((self.layer[i].size,1))
			delta = self.delta[i+1].reshape((1,self.layer[i+1].size))
			# print 'a',i,':',a[:5,0]
			# print 'delta',i,':',delta[0,:5]
			self.dW[i] = a.dot(delta)
			self.db[i] = self.delta[i+1]

		return self.dW, self.db, loss

	def gradient_check(self, x, y, h=1e-4):
		dx = 2*h
		# store the numerical gradients
		temp_W = self.W[0][0,0].copy()
		self.W[0][0,0] += h
		output = self.forward_pass(x)
		_, loss1 = self.compute_loss(output, y)
		
		# -s
		self.W[0][0,0] -= 2*h
		output = self.forward_pass(x)
		_, loss2 = self.compute_loss(output, y)

		dW = (loss1[0] - loss2[0])/dx

		self.W[0][0,0] = temp_W.copy()

		return dW

	def train(self, X, y, full_epoch=True, gradient_check=False, epoch=1000, reg=0.001, learning_rate=0.001, verbose=False):

		# loss = 0
		if full_epoch == True:
			N = X.shape[0]
		else:
			N = epoch
		for i in xrange(N):
			# forward pass
			output = self.forward_pass(X[i])
			# backward pass
			self.dW, self.db, loss = self.backward_pass(X[i], output, y[i])
			# verbose
			if verbose:
				print i
				print 'W:', self.W
				print 'layer_output:', self.layer_output
				print 'b:', self.b
				print 'output:', output
			# print i,'dW[0]:',self.dW[0][:5,:5]   # check that dW[0] is always 0
			# print i,'dout[0]:',self.dout[0] 
			# gradient check
			if gradient_check:
				num_dW = self.gradient_check(X[i], y[i])
				# print i
				# print 'diff:', (num_dW - self.dW[0][:,0])[:3]
				print 'num_dW:',num_dW
				print 'anal_dW:', self.dW[0][0,0]
			# update the parameters
			for j in xrange(self.layer_num-1):
				self.W[j] -= (learning_rate*self.dW[j] + reg/2*self.W[j])
				self.b[j] -= learning_rate * self.db[j]
			# print i, self.W[self.layer_num-2][:5,:5]
		return self.W, self.b

