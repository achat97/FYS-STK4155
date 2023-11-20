import numpy as np
from functions import *

"""
All the gradient descent methods here calculate gradient of the cost function,
which is the MSE with an L2-regularisation, using automatic differentiation (JAX)

"""

def GD(X,y,betainit,niterations,eta,lmbda=0,gamma=0):
    """
    Gradient descent with a constant learning rate and optional momentum

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        niterations (int): max iterations, i.e. steps
        eta (float): learning rate
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters

    """  

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    v = gamma*v-eta*g(beta)
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta


def GD_Adagrad(X,y,betainit,niterations,eta,lmbda=0,delta=1e-7,gamma=0):
    """
    Gradient descent with Adagrad and optional momentum

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        niterations (int): max iterations, i.e. steps
        eta (float): global learning rate
        delta (float): small value for numerical stability
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters

    """ 

    v = 0

    beta = np.copy(betainit)
    r = np.zeros(beta.shape)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    
                    r += g(beta)*g(beta)
                    v = gamma*v-(eta/(delta+np.sqrt(r)))*g(beta)
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta


def GD_RMSprop(X,y,betainit,niterations,eta,rho=0.9,delta=1e-6,lmbda=0,gamma=0):
    """
    Gradient descent with RMSProp and optional momentum

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        niterations (int): max iterations, i.e. steps
        eta (float): global learning rate
        rho (float): decay rate
        delta (float): small value for numerical stability
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters

    """ 

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    r = np.zeros(beta.shape)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    
                    r = rho*r+(1-rho)*g(beta)*g(beta)
                    v = gamma*v-(eta/(np.sqrt(delta+r)))*g(beta)
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta


def GD_ADAM(X,y,betainit,niterations,eta,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0):
    """
    Gradient descent with Adam

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        niterations (int): max iterations, i.e. steps
        eta (float): global learning rate
        rho1(float): decay rate for first moment
        rho2(float): decay rate for second moment
        delta (float): small value for numerical stability
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters

    """ 

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)

    r = np.zeros(beta.shape)
    s = np.zeros(beta.shape)


    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda) 
                    
                    s = rho1*s+(1-rho1)*g(beta)
                    r = rho2*r+(1-rho2)*g(beta)*g(beta)

                    shat = s/(1-rho1**(i+1))
                    rhat = r/(1-rho2**(i+1))

                    v = -eta*(shat/(delta+np.sqrt(rhat)))
                    
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta