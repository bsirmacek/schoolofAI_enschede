import tensorflow as tf
import numpy as np 

class Autoencoder: 
	def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):
		self.epoch = epoch #number of learning cycles 
		self.learning_rate = learning_rate #hyperparameter of the optimizer

		#define the input layer dataset:
		x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])

		#define weights and biases under a name scope
		with tf.name_scope('encode'):
			weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
			biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
			encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)
		with tf.name_scope('decode'):
			weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
			biases = tf.Variable(tf.zeros([input_dim]), name='biases')
			decoded = tf.matmul(encoded, weights) + biases

		self.x = x
		self.encoded = encoded
		self.decoded = decoded

		#reconstruction loss
		self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
		#choose the optimizer
		self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		#set up a saver to save model parameters as they are being learned
		self.saver = tf.train.Saver() 


	# for training the autoencoder:
	def train(self, data):
		num_samples = len(data)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer()) #starts a tensorflow session and initializes all variables
			for i in range(self.epoch): #iterates through the number of cycles defined in the constructor
				for j in range(num_samples):
					l, _ = sess.run([self.loss, self.train_op], feed_dict = {self.x: [data[j]]})
				if i % 10 == 0:
					print('epoch {0}: loss = {1}'.format(i, l)) #prints the reconstruction error once in every 10 cycles
					self.saver.save(sess, './model.ckpt') #saves the learned paramaters
			self.saver.save(sess, './model.ckpt')


	# for testing the model on data:
	def test(self, data):
		with tf.Session() as sess:
			self.saver.restore(sess, './model.ckpt') # loads the learned parameters
			hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data}) # reconstructs the input
		print('input', data)
		print('compressed', hidden)
		print('reconstructed', reconstructed)
		return reconstructed


