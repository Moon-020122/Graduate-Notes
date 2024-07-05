import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from lib.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


np.random.seed(1)


# GRADED FUNCTION: linear_function

def linear_function():
    np.random.seed(1)
    
    ### START CODE HERE ### (4 lines of code)  
    X = tf.constant(np.random.randn(3,1), name = "X")  
    W = tf.constant(np.random.randn(4,3), name = "W")  
    b = tf.constant(np.random.randn(4,1), name = "b")  
    Y = tf.add(tf.matmul(W,X),b)  
    ### END CODE HERE ###   
      
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate  
      
    ### START CODE HERE ###  
    sess = tf.Session()  
    result = sess.run(Y)  
    ### END CODE HERE ###   
    
    # close the session 
    sess.close()

    return result



def linear_function():

    np.random.seed(1)
    
    ### START CODE HERE ### (4 lines of code)  
    X = tf.constant(np.random.randn(3,1), name = "X")  
    W = tf.constant(np.random.randn(4,3), name = "W")  
    b = tf.constant(np.random.randn(4,1), name = "b")  
    Y = tf.add(tf.matmul(W,X),b)  
    ### END CODE HERE ###   
      
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate  
      
    ### START CODE HERE ###  
    sess = tf.Session()  
    result = sess.run(Y)  
    ### END CODE HERE ###   
    
    # close the session 
    sess.close()

    return result




# GRADED FUNCTION: cost

def cost(logits, labels):

    
    ### START CODE HERE ###   
      
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)  
    z = tf.placeholder(tf.float32, name = "z")  
    y = tf.placeholder(tf.float32, name = "y")  
      
    # Use the loss function (approx. 1 line)  
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)  
      
    # Create a session (approx. 1 line). See method 1 above.  
    sess = tf.Session()  
      
    # Run the session (approx. 1 line).  
    cost = sess.run(cost,feed_dict={z:logits,y:labels})  
      
    # Close the session (approx. 1 line). See method 1 above.  
    sess.close()  
      
    ### END CODE HERE ###  
    
    return cost

# GRADED FUNCTION: one_hot_matrix

def one_hot_matrix(labels, C):
    
    ### START CODE HERE ###  
      
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)  
    C = tf.constant(C, name = "C")  
      
    # Use tf.one_hot, be careful with the axis (approx. 1 line)  
    one_hot_matrix = tf.one_hot(labels, C, axis=0)  
      
    # Create the session (approx. 1 line)  
    sess = tf.Session()  
      
    # Run the session (approx. 1 line)  
    one_hot = sess.run(one_hot_matrix)  
      
    # Close the session (approx. 1 line). See method 1 above.  
    sess.close()  
      
    ### END CODE HERE ###  
    
    return one_hot

# GRADED FUNCTION: ones

def ones(shape):

    ### START CODE HERE ###  
      
    # Create "ones" tensor using tf.ones(...). (approx. 1 line)  
    ones = tf.ones(shape)  
      
    # Create the session (approx. 1 line)  
    sess = tf.Session()  
      
    # Run the session to compute 'ones' (approx. 1 line)  
    ones = sess.run(ones)  
      
    # Close the session (approx. 1 line). See method 1 above.  
    sess.close()  
      
    ### END CODE HERE ###  
    return ones  


# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)


# GRADED FUNCTION: create_placeholders

def create_placeholders(n_x, n_y):


    ### START CODE HERE ### (approx. 2 lines)  
    X = tf.placeholder(shape=[n_x, None],dtype=tf.float32)  
    Y = tf.placeholder(shape=[n_y, None],dtype=tf.float32)  
    ### END CODE HERE ###  
    
    return X, Y

def initialize_parameters():
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)  
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))  
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())  
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))  
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())  
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))  
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())  
    ### END CODE HERE ### 

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:  
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1  
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)  
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2  
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)  
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3,Z2) + b3  
    ### END CODE HERE ###  
    
    return Z3


# GRADED FUNCTION: compute_cost 

def compute_cost(Z3, Y):

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))  
    ### END CODE HERE ###  
    
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.random.set_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)  
    X, Y = create_placeholders(n_x, n_y)  
    ### END CODE HERE ###  
  
    # Initialize parameters  
    ### START CODE HERE ### (1 line)  
    parameters = initialize_parameters()  
    ### END CODE HERE ###  
      
    # Forward propagation: Build the forward propagation in the tensorflow graph  
    ### START CODE HERE ### (1 line)  
    Z3 = forward_propagation(X, parameters)  
    ### END CODE HERE ###  
      
    # Cost function: Add cost function to tensorflow graph  
    ### START CODE HERE ### (1 line)  
    cost = compute_cost(Z3, Y)  
    ### END CODE HERE ###  
      
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.  
    ### START CODE HERE ### (1 line)  
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)  
    ### END CODE HERE ###  
      
    # Initialize all the variables  
    init = tf.global_variables_initializer()  
  
    # Start the session to compute the tensorflow graph  
    with tf.Session() as sess:  
          
        # Run the initialization  
        sess.run(init)  
          
        # Do the training loop  
        for epoch in range(num_epochs):  
  
            epoch_cost = 0.                       # Defines a cost related to an epoch  
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set  
            seed = seed + 1  
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)  
  
            for minibatch in minibatches:  
  
                # Select a minibatch  
                (minibatch_X, minibatch_Y) = minibatch  
                  
                # IMPORTANT: The line that runs the graph on a minibatch.  
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).  
                ### START CODE HERE ### (1 line)  
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})  
                ### END CODE HERE ###  
                  
                epoch_cost += minibatch_cost / num_minibatches  
  
            # Print the cost every epoch  
            if print_cost == True and epoch % 100 == 0:  
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))  
            if print_cost == True and epoch % 5 == 0:  
                costs.append(epoch_cost)  
                  
        # plot the cost  
        plt.plot(np.squeeze(costs))  
        plt.ylabel('cost')  
        plt.xlabel('iterations (per tens)')  
        plt.title("Learning rate =" + str(learning_rate))  
        plt.show()  
  
        # lets save the parameters in a variable  
        parameters = sess.run(parameters)  
        print ("Parameters have been trained!")  
  
        # Calculate the correct predictions  
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))  
  
        # Calculate accuracy on the test set  
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
  
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))  
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))  
          
        return parameters  
    

parameters = model(X_train, Y_train, X_test, Y_test)