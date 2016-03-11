import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import nn
import pickle

csv_reader = csv.reader(open(r'C:\Users\Administrator\Desktop\mnist\train.csv','rb'))
header = csv_reader.next()
data = []
label = []
data_size = 1000 # 4000 best choice
for row in csv_reader:
	data.append(row[1:])
	label.append(row[0])
	if len(data) >= data_size:
		break

data = np.array(data,dtype=np.float)
label = np.array(label)
#print data[10][:]
# image = data[10][:].reshape((28,28))
# plt.imshow(image,cmap='gray')
# plt.show()

input_size = 28*28
output_size = 10
hidden_size = 500
learning_rate = 0.001 # 0.001

training_data = data[:data_size*0.7,:]
training_label = label[:data_size*0.7]
test_data = data[data_size*0.7:,:]
test_label = label[data_size*0.7:]

net = nn.TwoLayerNeuralNet(input_size, output_size, hidden_size)
loss_history = net.train(training_data[:data_size*0.7,:], training_label[:data_size*0.7], learning_rate)

n = 0
for i in xrange(test_label.shape[0]):
	result = net.predict(test_data[i,:])
	#print str(result),test_label[i]
	if str(result) == test_label[i]:
		n += 1

N = test_label.shape[0]
accuracy = n*1.0/test_label.shape[0]
print 'data_size:', data_size
print 'hidden_size:', hidden_size
print 'accuracy:', accuracy*100, '%'
#print 'loss_history:', loss_history
# plt.plot(xrange(loss_history.shape[0]), loss_history, 'r')
# plt.show()
pkl = open('data.pkl','wb')
W1 = net.W1
W2 = net.W2
b1 = net.b1
b2 = net.b2
pickle.dump(W1, pkl)
pickle.dump(W2, pkl)
pickle.dump(b1, pkl)
pickle.dump(b2, pkl)

pkl.close()