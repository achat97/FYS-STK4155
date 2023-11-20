import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import seaborn as sns
from functions import *




def FF_initialize(x,nlayers,nneurons,ncategories):
    """
    Initialise weight and biases for neural network using Glorot initialisation

    Args:
        x (ndarray): input data
        nlayers (int): number of hidden layers
        nneurons (int): number of neurons per hidden layer
        ncategories (int): number of neurons in the output layer

    Returns:
        w (ndarray): weights
        b (ndarray): biases
    """ 

    np.random.seed(2014)
    
    ninputs, nfeatures = x.shape

    w = np.empty(nlayers+1, dtype=object)
    b = np.empty(nlayers+1, dtype=object)

    if nlayers>0:
        w[0] = np.random.normal(0,(2/(nfeatures+nneurons))**2,size=(nfeatures, nneurons))
        b[0] = np.zeros(nneurons) + 0.01

        for i in range(1,nlayers):
            w[i] = np.random.normal(0,(1/nneurons)**2,size=(nneurons, nneurons))
            b[i] =  np.zeros(nneurons) + 0.01

        w[nlayers] = np.random.normal(0,(2/(ncategories+nneurons))**2,size=(nneurons, ncategories))
        b[nlayers] = np.zeros(ncategories) + 0.01

    else:
        w[0] = np.random.normal(0,(2/(ninputs+nfeatures))**2,size=(nfeatures,ncategories))
        b[0] = np.zeros(ncategories) + 0.01

    return w,b


def feed_forward(x,w,b,f):
    """
    Feeds the data to the next layer and calculates the output

    Args:
        x (ndarray): input data
        w (ndarray): weights
        b (ndarray): biases
        f (function): activation function

    Returns:
        a (ndarray): layer output
    """
 
    z = np.matmul(x,w)+b
    a = f(z)

    return a


def feed_forward_pass(x,w,b,nlayers,f_h,f_o):
    """
    Does one feedforward pass, i.e. feeds forward from input layer to output layer

    Args:
        x (ndarray): input data
        w (ndarray): weights
        b (ndarray): biases
        f_h (function): activation function for hidden layers
        f_o (function): activation function for output layer

    Returns:
        a (ndarray): ouput of all layers
    """

    a = np.empty(nlayers+2, dtype=object)
    a[0] = x
    
    for i in range(1,nlayers+1):
        a[i] = feed_forward(a[i-1],w[i-1],b[i-1],f_h)

    a[nlayers+1] = feed_forward(a[nlayers],w[nlayers],b[nlayers],f_o)

    return a


def backpropagation(x,t,w,b,nlayers,fh,fo,fh_derivative,fo_derivative,cost_derivative):
    """
    Backpropagation algorithm

    Args:
        x (ndarray): input data
        t (ndarray): target data
        w (ndarray): weights
        b (ndarray): biases
        nlayers (int): number of hidden layers
        f_h (function): activation function for hidden layers
        f_o (function): activation function for output layer
        fh_derivative (function): derivative of activation function for hidden layers
        fo_derivative (function): derivative of activation function for output layer
        cost_derivative (function): derivative of cost function

    Returns:
        w_grad (ndarray): derivative of cost wrt. weights
        b_grad (ndarray): derivative of cost wrt. bias
    """

    a = feed_forward_pass(x,w,b,nlayers,fh,fo)
    L = nlayers
    
    if L>0:
        error = np.empty(L+1, dtype=object)
        w_grad = np.empty(L+1, dtype=object)
        b_grad = np.empty(L+1, dtype=object)

        error[L] = fo_derivative(np.matmul(a[L],w[L])+b[L])*cost_derivative(t,a[L+1])

        for i in range(1,L+1):
            error[L-i] = np.matmul(error[L-i+1], w[L-i+1].T)*fh_derivative(np.matmul(a[L-i],w[L-i])+b[L-i])

        for i in range(L+1):
            w_grad[L-i] = np.matmul(a[L-i].T,error[L-i])
            b_grad[L-i] = np.sum(error[L-i], axis=0)

    else:

        error = np.empty(1, dtype=object)
        w_grad = np.empty(1, dtype=object)
        b_grad = np.empty(1, dtype=object)

        error[0] = fo_derivative(np.matmul(a[L],w[L])+b[L])*cost_derivative(t,a[L+1])

        w_grad[0] = np.matmul(a[0].T,error[0])
        b_grad[0] = np.sum(error[0], axis=0)

    return w_grad, b_grad


def FFNN(x,y,nlayers,nneurons,nepochs,nbatches,fo,fh,fo_derivative,fh_derivative,cost_derivative,lmbda=0):
    """
    Complete feedforward neural network training. Weights and biases are found using SGD with Adam.

    Args:
        x (ndarray): input data
        t (ndarray): target data
        nlayers (int): number of hidden layers
        nneurons (int): number of neurons per hidden layer
        nepochs (int): number of epochs
        nbatches (int): number of batches
        f_o (function): activation function for output layer
        f_h (function): activation function for hidden layers
        fo_derivative (function): derivative of activation function for output layer
        fh_derivative (function): derivative of activation function for hidden layers
        cost_derivative (function): derivative of cost function
        lmbda (float): hyperparameter

    Returns:
        w (ndarray): weights
        b (ndarray): biases
    """
    np.random.seed(2014)

    w,b = FF_initialize(x,nlayers,nneurons,1)
    n = x.shape[0]
    ind = np.arange(n)
    np.random.shuffle(ind)
    batch = np.array_split(ind,nbatches)

    rw = np.empty(len(w),dtype=object)
    rb = np.empty(len(b),dtype=object)

    sw = np.empty(len(w),dtype=object)
    sb = np.empty(len(b),dtype=object)

    rho1 = 0.9
    rho2 = 0.999
    eta = 0.001
    delta = 1e-7

    vw = np.empty(len(w),dtype=object)
    vb = np.empty(len(b),dtype=object)

    for i in range(len(w)):
        rw[i] = np.zeros(np.shape(w[i]))
        sw[i] = np.zeros(np.shape(w[i]))
        vw[i] = np.zeros(np.shape(w[i]))

        rb[i] = np.zeros(np.shape(b[i]))
        sb[i] = np.zeros(np.shape(b[i]))
        vb[i] = np.zeros(np.shape(b[i]))


    t=0
    for epoch in range(nepochs):
        for k in range(nbatches):

            xk = x[batch[k]]
            yk = y[batch[k]]

            dW, dB = backpropagation(xk,yk,w,b,nlayers,fo,fh,fo_derivative,fh_derivative,cost_derivative)


            t+=1
            for i in range(len(rw)):
                dW[i] += lmbda * w[i]

                sw[i] = rho1*sw[i]+(1-rho1)*dW[i]
                sb[i] = rho1*sb[i]+(1-rho1)*dB[i]
                
                
                rw[i] = rho2*rw[i]+(1-rho2)*dW[i]*dW[i]
                rb[i] = rho2*rb[i]+(1-rho2)*dB[i]*dB[i]

                swhat = sw[i]/(1-rho1**(t))
                rwhat = rw[i]/(1-rho2**(t))

                sbhat = sb[i]/(1-rho1**(t))
                rbhat = rb[i]/(1-rho2**(t))
                
                vw[i] = -eta*(swhat/(delta+np.sqrt(rwhat)))
                vb[i] = -eta*(sbhat/(delta+np.sqrt(rbhat)))


                w[i]+= vw[i]
                b[i]+= vb[i]

    return w,b            