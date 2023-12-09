from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense  
import matplotlib.pyplot as plt

np.random.seed(200)


def create_neural_network(nneurons,act,initialisation):
    """
    Sets up the neural network and its architecture

    Args:
        nneurons (list): list of neurons per hidden layer
        act (str): activation function - sigmoid,tanh,relu,elu
        initialisation (str): type of initilisation of the weights - glorot and he
        
    Returns:
        model (object): neural network
        opt (object): optimiser (Adam) of neural network 
    """

    model = Sequential()
    bias =  tf.constant_initializer(0.1)
   
    
    if initialisation.lower() == 'glorot':
        initializer = keras.initializers.GlorotNormal(seed=200)

    elif initialisation.lower() == 'he':
        initializer = keras.initializers.HeNormal(seed=200)

    model.add(Dense(nneurons[0],activation=act,kernel_initializer=initializer,bias_initializer=bias,input_dim=2))
    for i in range(1,len(nneurons)-1):
        model.add(Dense(nneurons[i],activation=act,kernel_initializer=initializer,bias_initializer=bias))

    model.add(Dense(1,activation='linear',kernel_initializer=initializer,bias_initializer=bias))

    opt = keras.optimizers.Adam()

    return model, opt



def g_trial(model,x,t,train):
    """
    Trial solution of the 1D heat equation
    
    Args:
        model: neural network
        x,t (tensor): input
        train (bool): True - training mode, False - inference mode
        
    Returns:
            (tensor): value of the trial function evaluated at the given x and t
    """

    X = tf.stack([x,t], axis=1)
    N = model(X,training=train)
    return tf.sin(np.pi*x)+x*(1.-x)*t*N



def loss(model,loss_obj,x,t):
    """
    Finds the derivatives of the trial and calculates the loss
    
    Args:
        model: neural network
        loss_obj (obj): object of the loss function 
         x,t (tensor): input
        
    Returns:
            (tensor): Loss of prediction
    """

    with tf.GradientTape() as tape2:
        tape2.watch(x)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            gtrial = g_trial(model,x,t,train=True)

        dg_dt=tape1.gradient(gtrial,t)
        dg_dx=tape1.gradient(gtrial,x)

    d2g_dx2=tape2.gradient(dg_dx,x)

    del tape1
    del tape2

    ypred = d2g_dx2-dg_dt
    
    return loss_obj(0.0,ypred)



def grad(model,loss_obj,x,t):
    """
    Calculates the gradient of the loss function with respect to the trainable variables
    
    Args:
        model: neural network
        loss_obj (obj): object of the loss function 
         x,t (tensor): input
        
    Returns:
            loss_value (tensor): Loss of prediction
            (tensor): gradient of loss
    """

    with tf.GradientTape() as tape:
        loss_value = loss(model,loss_obj,x,t)
    return loss_value, tape.gradient(loss_value,model.trainable_variables)



def fit(model,optimiser,x,t,batchsize,nepochs):
    """
    Trains the neural network to fit the 1D heat equation
    
    Args:
        model: neural network
        optimiser (object): learning rate optimiser for the neural network
        x,t (ndarray): input
        batchsize (int): size of each batchsize
        nepochs (int): number of epochs
        
    Returns:
            train_loss_results (list): average loss at each epoch
    """

    mse = tf.keras.losses.MeanSquaredError()
    x = x.reshape(-1,1)
    t = t.reshape(-1,1)

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)


    x_batch = tf.data.Dataset.from_tensor_slices(x).batch(batchsize)
    t_batch = tf.data.Dataset.from_tensor_slices(t).batch(batchsize)


    train_loss_results = []

    for epoch in range(nepochs):

        epoch_loss_avg = tf.keras.metrics.Mean()

        for x_, t_ in zip(x_batch, t_batch):

            loss_value, grads = grad(model,mse,x_,t_)
            optimiser.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)

        train_loss_results.append(epoch_loss_avg.result())

        print(f"Epoch: {epoch+1} – Loss: {epoch_loss_avg.result()}")

    return train_loss_results



def heat_analytical(x,t):
    """
    Analytical solution of the 1D heat equation
    
    Args:
        x,t (tensor): input
    Returns:
            (tensor): value of the function evaluated at the given x and t
    """  
    return tf.sin(np.pi*x)*tf.math.exp(-t*np.pi**2)

