# libraries
import tensorflow as tf
import numpy as np

# define architecture parameters
num_of_input_units  = 5
num_of_hidden_units = 4
num_of_output_units = 4

# network equations
x = tf.placeholder(tf.float32, shape=[1, num_of_input_units])
y = tf.placeholder(tf.float32, shape=[1, num_of_output_units])

weights_input_to_hidden = tf.Variable(tf.truncated_normal(shape=[num_of_input_units, num_of_hidden_units]))
biases_of_hidden_layer = tf.Variable(tf.zeros(shape=num_of_hidden_units))

weights_hidden_to_output = tf.Variable(tf.truncated_normal(shape=[num_of_hidden_units, num_of_output_units]))
biases_of_output_layer = tf.Variable(tf.zeros(shape=num_of_output_units))

# hidden layer output = sigmoid(Wx+b)
hidden_layer_output = tf.sigmoid(tf.add(tf.matmul(x, weights_input_to_hidden), biases_of_hidden_layer))

# output layer output = softmax(sigmoid(Wh+b))
network_output = tf.sigmoid(tf.add(tf.matmul(hidden_layer_output, weights_hidden_to_output), biases_of_output_layer))

# loss function
loss = tf.nn.softmax_cross_entropy_with_logits(network_output, y)

# train step
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# train the network