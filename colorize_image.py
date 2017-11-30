#encode pixel image to smaller dimension using autoencoder

import os# listdir
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def sigmoid(x, deriv=False):
    if(deriv==True):
        return x * (1 - x)
    
    return 1/(1 + np.exp(-x))

#%#%#%#%#%#% Neural Network #%#%#%#%#%#%
import tensorflow as tf
from math import floor
import time

class NeuralComputer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print('init: len(x) = {}, len(y) = {}'.format(len(self.x), len(self.y)))
        
        self.in_dim = len(x[0])
        self.out_dim = len(y[0])
        
    def Perceptron(self, tensor):
        #with tf.name_scope('softmax_linear'):
        #datatype as float16 to reduce RAM requirement and granularity in values regularize tensor
        V0 = tf.Variable(tf.truncated_normal([self.in_dim, 100]),
                         caching_device='/job:localhost/replica:0/task:0/device:GPU:0')
        b0 = tf.Variable(tf.truncated_normal([100]))
        l0 = tf.sigmoid(tf.matmul(tensor, V0) + b0)

        V1 = tf.Variable(tf.truncated_normal([100, 1000]))
        b1 = tf.Variable(tf.truncated_normal([1000]))
        l1 = tf.sigmoid(tf.matmul(l0, V1) + b1)

        V2 = tf.Variable(tf.truncated_normal([1000, 10000]))
        b2 = tf.Variable(tf.truncated_normal([10000]))
        l2 = tf.sigmoid(tf.matmul(l1, V2) + b2)

        V3 = tf.Variable(tf.truncated_normal([10000, 10000]))
        b3 = tf.Variable(tf.truncated_normal([10000]))
        l3 = tf.sigmoid(tf.matmul(l2, V3) + b3)

        V4 = tf.Variable(tf.truncated_normal([10000, 10000]))
        b4 = tf.Variable(tf.truncated_normal([10000]))
        l4 = tf.sigmoid(tf.matmul(l3, V4) + b4)
        
        V5 = tf.Variable(tf.truncated_normal([10000, 10000]))
        b5 = tf.Variable(tf.truncated_normal([10000]))
        l5 = tf.sigmoid(tf.matmul(l4, V5) + b5)

        V6 = tf.Variable(tf.truncated_normal([10000, 100]))
        b6 = tf.Variable(tf.truncated_normal([100]))
        l6 = tf.sigmoid(tf.matmul(l5, V6) + b6)
        
        weights = tf.Variable( tf.zeros([100, self.out_dim]),name='weights')
        biases = tf.Variable(tf.zeros([self.out_dim]),name='biases')

        logits = tf.nn.softmax(tf.matmul(l6, weights) + biases)
        
        return logits, weights, biases

    def init_placeholders(self, n_classes, batch_size):
        #init Tensors: fed into the model during training
        x = tf.placeholder(tf.float32, shape=(None, self.in_dim))
        y_ = tf.placeholder(tf.float32, shape=(batch_size, n_classes))

        #Neural Network Model
        y, W, b = self.Perceptron(x)

        return y, W, b, x, y_

    def train(self, test_x, in_str, batch_size=1000, training_epochs=10,learning_rate=.5,display_step=1):
        print('train: len(x) = {}, len(y) = {}'.format(len(self.x), len(self.y)))
        print('len(test_x):',len(test_x))
        #batch_size = len(test_x)
        test_size = batch_size* floor(len(self.x)/batch_size)

        #to verify accuracy on novel data
        acc_x = self.x[test_size - batch_size*2:]
        acc_y = self.y[test_size - batch_size*2:]
        print("acc_x:",len(acc_x), ' acc_y:',len(acc_y))
        
        print("len_train",int(test_size - batch_size*2))
        self.x = self.x[:test_size - batch_size*2]
        self.y = self.y[:test_size - batch_size*2]
        
        # Train W, b such that they are good predictors of y
        self.out_y, W, b, self.in_x, y_ = self.init_placeholders(self.out_dim, batch_size)

        # Cost function: Mean squared error
        cost = tf.reduce_sum(tf.pow(y_ - self.out_y, 2))/(batch_size)

        # Gradient descent: minimize cost via Adam Delta Optimizer (SGD)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=.99,epsilon=3e-08).minimize(cost)

        # Initialize variables and tensorflow session
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)

        start_time = time.time()
        print_time = True
        for i in range(training_epochs):
            j=0
            while j < len(self.x):
                start = j
                end = j + batch_size
                
                self.sess.run([optimizer, cost], feed_dict={self.in_x: self.x[start:end],
                                                            y_: self.y[start:end]})
                j += batch_size
            # Display logs for epoch in display_step
            if (i) % display_step == 0:
                if print_time:
                    print_time = False
                    elapsed_time = time.time() - start_time
                    print('Predicted duration of this session:',(elapsed_time*training_epochs//60) + 1,'minute(s)')
                cc = self.sess.run(cost, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})
                print("Training step: {} || cost= {}".format(i,cc))
                        
        print("\nOptimization Finished!\n")
        training_cost = self.sess.run(cost, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})
        print("Training cost=",training_cost,"\nW=", self.sess.run(W)[:1],"\nb=",self.sess.run(b),'\n')
        correct_prediction = tf.equal(tf.argmax(self.out_y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy for predictions of {}'.format(in_str),
                self.sess.run(accuracy, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})*100,'%')
        
        #str(self.sess.run(accuracy, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})*100//1) + ' %'

    def save(self, in_str):
        self.saver.save(self.sess, in_str)

    def load(self, graph):
        #out_y, W, b, in_x, y_ = self.init_placeholders(self.out_dim, batch_size)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(graph + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        
    def predict(self, test_x):
        predictions = []
        for matrix in test_x:
            predictions.append(self.sess.run(self.out_y, feed_dict={self.in_x:matrix}))

        self.sess.close()
        return predictions

    def max_of_predictions(self, predictions):
        out_arr = []
        for pred in predictions:
            #print('\n========')
            _max = [0, 0]# [index, value]
            for matrix in pred:
                for i, vect in enumerate(matrix):
                    if _max[1] <  vect:
                        _max[1] = vect
                        _max[0] = i
                    #print('{}::{}'.format(i,vect))
                #print(':MAX:', _max[0], _max[1])
            out_arr.append(_max[0])

        #indecies of max values in one-hot arrays
        return out_arr

class ImageTensor():
    def __init__(self, pixel_data):
        self.raw = pixel_data
        self.make_gray_scale()
        self.color_decomposition()
        self.combine_color_rows()
        

    def show(self, gray_scale=False):
        if not gray_scale:
            plt.imshow(self.raw, interpolation="none")
            plt.show()
        else:
            plt.imshow(self.gray_data, interpolation="none")
            plt.show()

    def make_gray_scale(self):
        height = len(self.raw)
        length = len(self.raw[0])

        self.gray_data = []
        for row in range(height):
            self.gray_data.append([])
            for col in range(length):
                self.gray_data[row].append([])

                average = (1/3) * np.log(25 + 0.299*self.raw[row][col][0] + 0.587*self.raw[row][col][1] + 0.114*self.raw[row][col][2])
                self.gray_data[row][col] = [average, average, average]
                '''
                self.gray_data[row][col] = [0.299 * self.raw[row][col][0],
                                            0.587 * self.raw[row][col][1],
                                            0.114 * self.raw[row][col][2]]
                '''
    def color_decomposition(self):
        red_channel = []
        blue_channel = []
        green_channel = []
        for i in range(len(self.raw)):
            red_channel.append(self.raw[i].T[0])
            blue_channel.append(self.raw[i].T[1])
            green_channel.append(self.raw[i].T[2])
        

        self.color_matrix = [red_channel,
                             blue_channel,
                             green_channel]

        self.gray_vector = []
        for i in range(len(self.gray_data)):
            self.gray_vector.append(np.array(self.gray_data[i]).T[0])

    def convert_channels_to_image(self, matrix):#self.unraveled_colors
        height = len(self.raw)
        length = len(self.raw[0])

        pixel_data = [[]]
        rows_appended = 0
        columns_appended = 0
        for i in range(len(matrix[0])):
            columns_appended += 1
            pixel_data[rows_appended].append([matrix[0][i],
                                              matrix[1][i],
                                              matrix[2][i]])

            if columns_appended == length:
                if rows_appended % 2 == 1:
                    pixel_data[rows_appended] = pixel_data[rows_appended][::-1]
                rows_appended += 1
                columns_appended = 0
                pixel_data.append([])

        return np.asarray(pixel_data[:-1])       

    def combine_color_rows(self):
        self.red_channel = []
        self.blue_channel = []
        self.green_channel = []

        for i, row in enumerate(self.color_matrix[0]):
            if i % 2 == 0:
                for elem in row:
                    self.red_channel.append(elem)
            else:
                for elem in row[::-1]:
                    self.red_channel.append(elem)
                        
        for i, row in enumerate(self.color_matrix[1]):
            if i % 2 == 0:
                for elem in row:
                    self.blue_channel.append(elem)
            else:
                for elem in row[::-1]:
                    self.blue_channel.append(elem)

        for i, row in enumerate(self.color_matrix[2]):
            if i % 2 == 0:
                for elem in row:
                    self.green_channel.append(elem)
            else:
                for elem in row[::-1]:
                    self.green_channel.append(elem)
                
        self.unraveled_colors = [self.red_channel, self.blue_channel, self.green_channel]

        gray_temp = []
        for i, row in enumerate(self.gray_vector):
            if i % 2 == 0:
                for elem in row:
                    gray_temp.append(elem)
            else:
                for elem in row[::-1]:
                    gray_temp.append(elem)

        self.gray_vector = gray_temp


#}+{-}+{-}+{-}+{-}+{-}+{-}+{-}+{-}+{-}+{}}

file_names = []
for filename in os.listdir("./Flowers/"):
    file_names.append(filename)

datum = []
size_of_input = 5
print("Initializing image data")
for i in range(size_of_input):
    if i % 2 == 0:
        print("--> {}%".format(i*100/size_of_input))
    im = Image.open('./Flowers/' + file_names[i])
    datum.append(ImageTensor(np.asarray(im)))
    #plt.imshow(data, interpolation="none")
    #plt.show()

print("Making training arrays")
train_x = []
train_y_red = []
train_y_green = []
train_y_blue = []
for ia in datum:
    train_x.append(ia.gray_vector)
    train_y_red.append(ia.red_channel)
    train_y_green.append(ia.green_channel)
    train_y_blue.append(ia.blue_channel)

test_x = train_x[3*len(train_x)//5:]
test_y_red = train_y_red[3*len(train_y_red)//5:]
test_y_green = train_y_green[3*len(train_y_green)//5:]
test_y_blue = train_y_blue[3*len(train_y_blue)//5:]

train_x = train_x[:3*len(train_x)//5]
train_y_red = train_y_red[:3*len(train_y_red)//5]
train_y_green = train_y_green[:3*len(train_y_green)//5]
train_y_blue = train_y_blue[:3*len(train_y_blue)//5]

print('\nInitializing Neural Network ...')
red_network = NeuralComputer(train_x, train_y_red)

red_network.train(test_y_red,"Red Channel",
                  training_epochs=5, learning_rate=.5,
                  batch_size=1,display_step=1)


green_network = NeuralComputer(train_x, train_y_green)
green_network.train(test_y_green,"Green Channel",
                    training_epochs=2, learning_rate=.5,
                    batch_size=1,display_step=1)

blue_network = NeuralComputer(train_x, train_y_blue)
blue_network.train(test_y_blue,"Blue Channel",
                  training_epochs=2, learning_rate=.5,
                  batch_size=1,display_step=1)

#ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[202500,10000]

### SHOW PREDICTIONS
# mat = it.convert_channels_to_image(it.unraveled_colors) #[r, g, b]
# plt.imshow(mat)
# plt.show()
'''


#Au = AutoEncoder(model['printing'], scale=0.5)

training_words = [inc.description for inc in all_incidents[:200]]

# feeding in normalized vectors
training_vects = []
for word in training_words:
    vect = model[word]
    total = 0  
    for num in vect:
        total += num ** 2
        
    mu = np.sqrt(total)/len(vect)
    vect -= mu
    vect = sigmoid(vect)
    training_vects.append(vect)
    
Au.train(training_vects, epochs=10, display_step = 2,learning_rate=0.05)
'''
