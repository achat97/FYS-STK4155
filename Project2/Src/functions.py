import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from random import random, seed

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



def CostOLS(y,X,beta):
    return np.sum((y-X @ beta)**2)


def CostRidge(y,X,beta,lmbda):
    return np.sum((y-X @ beta)**2)+lmbda*X.shape[0]*np.sum((beta)**2)

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


def step_length(t,t0,t1):
    return t0/(t+t1)