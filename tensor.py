import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from ceps_utils_tensor import read_train_ceps, read_test_ceps
from config import TRAIN_DATASET_DIR,TEST_DATASET_DIR,GENRE_LIST
import numpy as np
import random

genre_list = GENRE_LIST

print (" Starting classification \n")
print (" Classification running ... \n") 
X_train, y_train = read_train_ceps(genre_list)    
X_test, y_test = read_test_ceps(genre_list)

n_node_hl1 = 800
n_node_hl2 = 800
n_node_hl3 = 800

n_classes = len(genre_list)

batch_size = 50

z = []

for i in y_train:
	a = np.zeros(n_classes)
	a[i] = 1
	z.append(list(a))
y_train = np.array(z)

w = []
for i in y_test:
	a = np.zeros(n_classes)
	a[i] = 1
	w.append(list(a))
y_test = np.array(w)

# data = []
# X_train = list(X_train)
# y_train = list(y_train)
# for i in range(len(X_train)):
# 	data.append([(X_train[i]),(y_train[i])])
# random.shuffle(data)
# print(data[0])
# data = np.array(data)

# X_train = np.array(data[:,0])
# y_train = np.array(data[:,1])

# print(type(X_train))
# print(type(y_train))
# print(X_train)
# print(y_train)
permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
y_train = y_train[permutation]
print (len(X_train),len(y_train),len(X_train[0]))
#print (y_train[:50,:]) 

#sys.exit(1)

#print(len(X_train))



x = tf.placeholder('float',[None,len(X_train[0])])
y = tf.placeholder('float',[None,len(y_train[0])])

def neural_network_model(data):

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(X_train[0]),n_node_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_node_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl1,n_node_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_node_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl2,n_node_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_node_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl3,n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']) , hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']) , hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
	
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 100
	print ("hey")
	with tf.Session()  as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i = 0
			while i < len(X_train):
				start = i
				end = i+batch_size
				batch_x = np.array(X_train[start:end])
				batch_y = np.array(y_train[start:end])
				# print(type(batch_x))
				# print(type(batch_y))
				# print(len(batch_x))
				# print(len(batch_y))
				#print(batch_x)
				#print(batch_y)
				_, c = sess.run([optimizer,cost], feed_dict = {x: batch_x, y:batch_y})
				epoch_loss += c
				i += batch_size
				#print (start,end)
			if epoch%10==0:
			    print('Epoch',epoch,' completed out of ',hm_epochs,' loss: ',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy: ',accuracy.eval({x:X_test, y:y_test}))


train_neural_network(x)


