import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from random import random, seed
from jax import grad

def Design(x,pol):
    """
   Constructs design the matrix for the input data for a given polynomial degree

    Args:
        x (ndarray): x-values
        pol (int): polynomial degree

    Returns:
        X (ndarray): design matrix
    """    

    X = np.zeros((len(x[:,0]),pol+1))
    for i in range(pol+1):
        X[:,i] = x[:,0]**i
    return X


def OLS(X, z):
    """
    Calculates beta-parameters using ordinary least square method

    Args:
        X (ndarray): design matrix
        z (ndarray): target values

    Returns:
        beta (ndarray): beta-parameters

    """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z
    return beta


def Ridge(X,z,lam):

    """
    Calculates beta-parameters using Ridge regression

    Args:
        X (ndarray): design matrix
        z (ndarray): target values
        lambda (float): hyperparameter

    Returns:
        beta (ndarray): beta-parameters

    """
    beta = np.linalg.pinv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ z

    return beta


def MSE(z,zpred):
    """
    Calculates the mean square error

    Args:
        z (ndarray): target data
        zpred (ndarray): predicted target data

    Returns:
        Error (float): mean square error of model

    """
    return np.mean((z-zpred)**2)


def R2(z,zpred):
    """
    Calculates the R2-score

    Args:
        y (ndarray): target data
        ypred (ndarray): predicted target data

    Returns:
        Score (float): R2-score of model

    """
    return 1 - np.sum((z - zpred)**2) / np.sum((z - np.mean(z)) ** 2)


def gradient(X,y,beta,lmbda):
    """
    Creates a function of the derivative of the MSE with L2-regularisation cost function using JAX

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        beta (ndarray): regression parameters
        lmbda (float): hyperparameter

    Returns:
        grad (function): derivative of the cost funciton
    """
    cost = lambda beta: np.mean((y-X @ beta)**2)+lmbda*np.mean((beta)**2)
    return grad(cost)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def identity(x):
    return x


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


def identity_derivative(x):
    return np.ones(np.shape(x))


def RELU(x):
    return np.where(x > np.zeros(x.shape), x, np.zeros(x.shape))


def RELU_derivative(x):
    return np.where(x > 0, 1, 0)


def LRELU(x):
    delta = 10e-4
    return np.where(x > np.zeros(x.shape), x, delta * x)


def LRELU_derivative(x):
    delta = 10e-4
    return np.where(x > 0, 1, delta)


def mse(z,zpred):
    return 1/2 * (zpred-z)**2


def mse_derivative(z,zpred):
    return (zpred-z)


def crossentropy(z,zpred):
   return -(z*np.log(zpred+1e-10) + (1-z)*np.log(1-zpred+1e-10))


def crossentropy_derivative(z,zpred):
    return (zpred-z)/((zpred+1e-10)*(1-zpred+1e-10))




