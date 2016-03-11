import numpy as np
import nn
import csv as csv
import pickle
import matplotlib.pyplot as plt

csv_reader = csv.reader(open(r'C:\Users\Administrator\Desktop\mnist\train.csv','rb'))
header = csv_reader.next()
data = []
label = []
data_size = 30000 # 4000 best choice
for row in csv_reader:
	data.append(row[1:])
	label.append(row[0])
	if len(data) >= 40000:
	 	break

data = np.array(data,dtype=np.float)
label = np.array(label)
#print data[10][:]
# image = data[10][:].reshape((28,28))
# plt.imshow(image,cmap='gray')
# plt.show()

input_size = 28*28
output_size = 10
hidden_size = 5
learning_rate = 0.001 # 0.001

training_data = data[:data_size,:]
training_label = label[:data_size]
test_data = data[data_size*0.7:,:]
test_label = label[data_size*0.7:]

net1 = nn.NeuralNet(str(input_size) + ',500,'  + str(output_size))
# W, b = net1.train(training_data, training_label, learning_rate=0.001, gradient_check=True, reg=0)

# net1 = nn.NeuralNet('3,5,3')
# output = net1.forward_pass(np.array([0.1,3,2]))
# print output
# real_output, loss = net1.compute_loss(output, 1)
# print 'real_output:', real_output
# print 'loss:', loss
# delta = net1.compute_error(output, 1)
# print delta
# W,b = net1.train(training_data, training_label, full_epoch=True, verbose=False)
# net2 = nn.TwoLayerNeuralNet(input_size, output_size, hidden_size)
# x = np.array([1,2,3])
# y = np.array([0])



# pkl_file = open('data.pkl', 'rb')
# net1.W[0] = pickle.load(pkl_file)
# net1.W[1] = pickle.load(pkl_file)
# net1.b[0] = pickle.load(pkl_file)
# net1.b[1] = pickle.load(pkl_file)

# net2.W1 = net1.W[0]
# net2.W2 = net1.W[1]
# net2.b1 = net1.b[0]
# net2.b2 = net1.b[1]

# output = net1.forward_pass(training_data[0])
# net1_dW, net1_db, net1_dout = net1.compute_dL(output, training_label[1])
# loss, a1, a2 = net2.forward_pass(training_data[0], training_label[0])
# dW1,dW2,db1,db2 = net2.backward_pass(training_data[1], training_label[1], a1, a2)
# # print dW1, dW2, db1, db2

# print 'difference:'
# print net1_dW[1] - dW2

# test
test_data = data[30000:,:]
test_label = label[30000:]
accuracy_history = []
net1 = nn.NeuralNet(str(input_size) + ',500,'  + str(output_size))
for j in xrange(30):
	training_data = data[1000*j:1000*(j+1),:]
	training_label = label[1000*j:1000*(j+1)]
	
	net1_W, net1_b = net1.train(training_data, training_label, learning_rate=0.001, reg=0.001)

	n = 0
	for i in xrange(test_data.shape[0]):
		output = net1.forward_pass(test_data[i])
		result = output.argmax()
		if str(result) == test_label[i]:
			n += 1
		# print i, result, test_label[i]
	accuracy = n*1.0/test_label.shape[0] * 100
	accuracy_history.append(accuracy)
	# print 'data_size:', data_size
	# print 'accuracy:', accuracy, '%','\n'
plt.plot(xrange(30), accuracy_history, 'r')
plt.show()