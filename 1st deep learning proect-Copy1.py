#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import library


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage




get_ipython().run_line_magic('matplotlib', 'inline')


# from urllib.request import urlopen
# import zipfile
# 
# print ("downloading with urllib2...please wait...")
# 
# # download data
# url_utils = 
# url_data = 
# 
# utils = urlopen(url_utils) 
# data = urlopen(url_data)
# 
# file_utils = utils.read() 
# dataset = data.read() 
# 
# # save data to file
# with open("lr_utils.py", "wb") as code:     
#     
#     code.write(file_utils)
# 
# with open("datasets.zip", "wb") as code:     
#     code.write(dataset)
# 
# # unzip datasets
# with zipfile.ZipFile("datasets.zip","r") as zip_ref:
#     zip_ref.extractall("")

# In[ ]:





# In[ ]:





# In[ ]:





# # loading data set 

# In[7]:



def load_dataset():
 train_dataset = h5py.File('train_catvnoncat.h5', "r")
 train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
 train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

 test_dataset = h5py.File('test_catvnoncat.h5', "r")
 test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
 test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

 classes = np.array(test_dataset["list_classes"][:]) # the list of classes
 
 train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
 test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
 
 return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[ ]:





# In[8]:


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# In[9]:


train_set_x_orig.shape


# In[10]:


# Example of a picture
index =208

plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


# #####     
# 
#     

# In[11]:


### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px =test_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# In[ ]:





# In[12]:


# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten= train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


# In[ ]:





# In[13]:


train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255.


# In[27]:


train_set_x.shape


# In[9]:


import PIL
print('Pillow Version:', PIL.__version__)


# In[10]:


from PIL import Image


# In[ ]:





# # all function name

# ## 1. Sigmoid function(z)
# ## 2. initialize_parameters:
#     initialiaze_with_zeros(dim)
# ## 3.propogate(w,b,X,Y)
# ##  4.optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False)
# ## 5.pedict(w,b,X)
# ## 6.model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False)

# In[ ]:





# ## 2 - Overview of the Problem set ##
# 
# **Problem Statement**: You are given a dataset ("data.h5") containing:
#     - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#     - a test set of m_test images labeled as cat or non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
# 
# You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.
# 
# Let's get more familiar with the dataset. Load the data by running the following code.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Loading data

# In[11]:


import PIL
print('Pillow Version:', PIL.__version__)


# In[12]:


from PIL import Image


# In[ ]:





# In[ ]:





# In[13]:


# GRADED FUNCTION: sigmoid

def sigmoid(z):


    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###
    
    return s


# In[14]:


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


# In[ ]:





# In[ ]:





# # Algrithm

# ## 3 - General Architecture of the learning algorithm ##
# 
# It's time to design a simple algorithm to distinguish cat images from non-cat images.
# 
# You will build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**
# 
# <img src="images/LogReg_kiank.png" style="width:650px;height:400px;">
# 
# **Mathematical expression of the algorithm**:
# 
# For one example $x^{(i)}$:
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$
# 
# The cost is then computed by summing over all training examples:
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$
# 
# **Key steps**:
# In this exercise, you will carry out the following steps: 
#     - Initialize the parameters of the model
#     - Learn the parameters for the model by minimizing the cost  
#     - Use the learned parameters to make predictions (on the test set)
#     - Analyse the results and conclude

# ### sigmoid function

# ## 4 - Building the parts of our algorithm ## 
# 
# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)
# 
# You often build 1-3 separately and integrate them into one function we call `model()`.
# 
# ### 4.1 - Helper functions
# 
# **Exercise**: Using your code from "Python Basics", implement `sigmoid()`. As you've seen in the figure above, you need to compute $sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$ to make predictions. Use np.exp().

# In[15]:


train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# In[16]:


import numpy as np

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s


# In[17]:


sigmoid(np.array([0,2,3,4,5,]))


# In[18]:


sigmoid(np.array([-10000,2000000,300000000,40000000,5000000,]))


# In[ ]:





# In[19]:


print('sigmoid of ([0,2,3,4,5,10000,-100000]) is \n=',sigmoid(np.array([0,2,3,4,5,10000,-100000])))


# In[ ]:





# In[20]:


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T,X)+b)                                   # compute activation
    cost = -1./m *np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))                            # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1./m*np.dot(X,(A-Y).T)
    db = 1./m*np.sum(A-Y)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


# In[ ]:





# In[21]:


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w-learning_rate * dw
        b = b-learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# In[ ]:





# In[22]:


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0,i] > 0.5 :
            Y_prediction[0,i] =1
        else:
            Y_prediction[0,i] =0
                
            
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# In[ ]:





# In[23]:


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    ean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[ ]:





# In[24]:


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# In[25]:


# Example of a picture that was wrongly classified.
index = 2
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")


# In[26]:


index = 14

plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]))


        
    


# In[27]:


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[28]:


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# In[30]:


## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "cat_1.jpg"   # change this to the name of your image file 
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255.
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### initilizing parameters

# In[ ]:


def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b


# In[ ]:


dim=3
w,b=initialize_with_zeros(dim)
print('w = '+ str(w))
print('b = '+ str(b))


# In[ ]:


print(w.shape)


# In[ ]:





# ## forward and backward propogation

# Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.
# 
# **Exercise:** Implement a function `propagate()` that computes the cost function and its gradient.
# 
# **Hints**:
# 
# Forward Propagation:
# - You get X
# - You compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
# - You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
# 
# Here are the two formulas you will be using: 
# 
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

# In[ ]:


def propogate(w,b,X,Y):
    
    import numpy as np
    
    m=X.shape[1]
    # forward prop
    
    A=sigmoid(np.dot(w.T,X)+b)
    cost=-1./m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    
    #back prp
    dw=1./m*np.dot(X,(A-Y).T)
    db=1./m*np.sum((A-Y))
    
    grads={'dw':dw,
           'db':db
        }
    return grads,cost


# In[ ]:


import numpy as np
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propogate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


# Expected Output:
# 
#         dw 	[[ 0.99845601] [ 2.39507239]]
#         db 	0.00145557813678
#         cost 	5.801545319394553 
# 
# 

# ## optimize

# Optimization
# 
#     You have initialized your parameters.
#     You are also able to compute a cost function and its gradient.
#     Now, you want to update the parameters using gradient descent.
# 
# Exercise: Write down the optimization function. The goal is to learn ww and bb by minimizing the cost function JJ. For a parameter θθ, the update rule is θ=θ−α dθθ=θ−α dθ, where αα is the learning rate.
# 

# In[10]:



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
  
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propogate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w-learning_rate * dw
        b = b-learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# In[11]:


params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


# Expected Output:
# 
# 
#                     w 	[[ 0.19033591] [ 0.12259159]]
#                     b 	1.92535983008
#                     dw 	[[ 0.67752042] [ 1.41625495]]
#                     db 	0.219194504541 

# **Exercise:** The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. There are two steps to computing predictions:
# 
# 1. Calculate $\hat{Y} = A = \sigma(w^T X + b)$
# 
# 2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). 

# In[ ]:


def predict(w,b,X):
    m=X.shape[1]
    y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    
    
    
    A=sigmoid(np.dot(w.T,X)+b)
    
    
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            y_prediction[0,i]=1
        else:
            y_prediction[0,i]=0
                         
                    
    return y_prediction                      
                         
                             


# In[19]:



import numpy as np
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))


# In[ ]:





# In[ ]:





# ## 5 - Merge all functions into a model ##
# 
# You will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.
# 
# **Exercise:** Implement the model function. Use the following notation:
#     - Y_prediction_test for your predictions on the test set
#     - Y_prediction_train for your predictions on the train set
#     - w, costs, grads for the outputs of optimize()

# In[21]:


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    # initialize with parameters
    w,b=initialize_with_zeros(x_train.shape[0])
    
    
    # Gadient descent
    
    parameters,grads,costs=optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w=parameters['w']
    b=parameters['b']
    
    #prediction
    y_prediction_test=predict(w,b,X_test)
    y_prediction_train=predict(w,b,X_train)
    
    
    #print train test error
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# In[3]:


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# Expected Output:
#     
#     
#     
# 
#                         Cost after iteration 0 	0.693147
#                         $\vdots$
# 
#                         $\vdots$
#                         Train Accuracy 	99.04306220095694 %
#                         Test Accuracy 	70.0 % 

# In[12]:


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[ ]:





# In[ ]:


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# In[ ]:




