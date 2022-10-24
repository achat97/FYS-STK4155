from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.utils import resample
from imageio import imread
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split,KFold
from random import random, seed
from sklearn import linear_model




def FrankeFunction(x,y):
    """
    Calculates the function values of the Franke function given x and y

    Args:
        x (ndarray): x-values
        y (ndarray): y-values

    Returns:
        f(x,y) (ndarray): function values
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4




def Data(N,z_tuning):

    """
    Generates x,y and z values

    Args:
    N (int): number of datapoints
    z_tuning (float): tuning parameter for noise

    Returns:

        z (ndarray): Franke function values over meshgrid
    """

    np.random.seed(2018)

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y) + z_tuning*np.random.randn(N, N)

    return x,y,z




def Data_terrain(tifFile,N,Nx,Ny):
    """
    Takes terrain data and return a cropped version of it in addition to producing random x and y data

    Args:
    tifFile (tif): tif-file with terrain data
    N (int): size of cropped version
    Nx (int): starting point (x-direction) in the terrain data for cropped version
    Ny (int): starting point (y-direction) in the terrain data for cropped version

    Returns:
    terrainData (ndarray): full terrain data
    x (ndarray): x-values over meshgrid
    y (ndarray): y-values over meshgrid
    z (ndarray): cropped terrain data
    """

    terrainData = np.array(imread(tifFile))
    size = N
    ylen, xlen = np.shape(terrainData)
    xregion = Nx
    yregion = Ny
    z = terrainData[xregion : xregion + size, yregion : yregion + size]

    np.random.seed(2018)
    x = np.sort(np.random.rand(size))
    y = np.sort(np.random.rand(size))
    x, y = np.meshgrid(x, y)

    return terrainData,x,y,z


def create_X(x, y, n):
    """
    Constructs design matrix

    Args:
        x (ndarray): x-values over meshgrid
        y (ndarray): x-values over meshgrid
        n (int): degree

    Returns:
        X (ndarray): design matrix

    """

    if len(x.shape)>1:
        x = np.ravel(x)    #flatten x
        y = np.ravel(y)    #flatten y

    N = len(x)
    l = int((n+1)*(n+2)/2)		#number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X




def features(matrix,degree):
    """
    Slices design matrix

    Args:
        matrix (ndarray): design matrix
        degree (int): degree of polynomial

    Returns:
        X (ndarray): design matrix for the given degree of polynomial

    """
    l = int((degree + 1) * (degree + 2) / 2)

    return matrix[:,:l]




def centering(Xtrain,Xtest,ztrain,ztest):
    """
    Zero centers design matrix and target values with training mean

    Args:
        Xtrain (ndarray): predictor training data
        Xtest (ndarray): predictor test data
        ztrain (ndarray): target training data
        ztrain (ndarray): target test data

    Returns:
        Arrays (ndarray): scaled training and test data
    """
    X_scaler = np.mean(Xtrain, axis=0)
    X_train_scaled = Xtrain - X_scaler
    X_test_scaled = Xtest - X_scaler

    z_scaler = np.mean(ztrain, axis=0)
    z_train_scaled = ztrain - z_scaler
    z_test_scaled = ztest - z_scaler

    return X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled




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

def Lasso(X,z,lam):


    """
    Creates a lasso regression model

    Args:
        X (ndarray): design matrix
        z (ndarray): target values
        lambda (float): hyperparameter

    Returns:
        Model : returns lasso regression model

    """
    RegLasso = linear_model.Lasso(fit_intercept=False, max_iter=100,alpha=lam).fit(X, z)

    return RegLasso



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



def bootstrap(Xtrain,Xtest,ztrain,ztest,B,method,lam):

    """
    Generating several predictions using same model by resampling training data (nonparemtric bootstrap method)

    Args:
        Xtrain (ndarray): predictor training data
        Xtest (ndarray): predictor test data
        ztrain (ndarray): target training data
        ztrain (ndarray): target test data
        B (int): number of bootstraps
        method (string): regression method (OLS, lasso or ridge)

    Returns:
        zpred (ndarray): matrix with a unique prediction at each column


    """

    if method.lower() == "ols":

        zpred = np.empty((len(ztest), B))
        for i in range(B):
            X_, z_ = resample(Xtrain, ztrain)
            beta = OLS(X_, z_)
            zpred[:, i] = (Xtest@beta).ravel()


    elif method.lower() == "ridge":

        zpred = np.empty((len(ztest), B))
        for i in range(B):
            X_, z_ = resample(Xtrain, ztrain)
            beta = Ridge(X_, z_,lam)
            zpred[:, i] = (Xtest@beta).ravel()


    elif method.lower() == "lasso":
        zpred = np.empty((len(ztest), B))
        for i in range(B):
            X_, z_ = resample(Xtrain, ztrain)
            model =  Lasso(X_,z_,lam)
            zpred[:, i] = model.predict(Xtest).ravel()

    return zpred





def crossValidation(X,z,k,method,lam=0):


    """
    Calculate mean square error of model using k-fold cross validation

    Args:
        X (ndarray): design matrix
        z (ndarray): target data
        k (int): number of folds
        method (string): regression method (OLS, lasso or ridge)

    Returns:
        error (float): mean square error
        error2 (float): R2-score

    """

    if method.lower() == "ols":

        kfold = KFold(n_splits = k)
        error = np.zeros(k)
        errorR2 = np.zeros(k)

        i=0
        for train_index, test_index in kfold.split(X):
            Xtrain = X[train_index,:]
            ztrain = z[train_index]

            Xtest = X[test_index,:]
            ztest = z[test_index]

            Xtrain_scaled,Xtest_scaled,ztrain_scaled,ztest_scaled = centering(Xtrain,Xtest,ztrain,ztest)


            beta = OLS(Xtrain_scaled, ztrain_scaled)
            zpred = (Xtest_scaled@beta).ravel()

            error[i] = MSE(ztest_scaled.ravel(),zpred)
            errorR2[i] = R2(ztest_scaled.ravel(),zpred)
            i += 1




    elif method.lower() == "ridge":

        kfold = KFold(n_splits = k)
        error = np.zeros(k)
        errorR2 = np.zeros(k)

        i=0
        for train_index, test_index in kfold.split(X):
            Xtrain = X[train_index,:]
            ztrain = z[train_index]

            Xtest = X[test_index,:]
            ztest = z[test_index]

            Xtrain_scaled,Xtest_scaled,ztrain_scaled,ztest_scaled = centering(Xtrain,Xtest,ztrain,ztest)

            beta = Ridge(Xtrain_scaled, ztrain_scaled,lam=lam)
            zpred = (Xtest_scaled@beta).ravel()

            error[i] = MSE(ztest_scaled.ravel(),zpred)
            errorR2[i] = R2(ztest_scaled.ravel(),zpred)
            i += 1


    elif method.lower() == "lasso":

        kfold = KFold(n_splits = k)
        error = np.zeros(k)
        errorR2 = np.zeros(k)

        i=0
        for train_index, test_index in kfold.split(X):
            Xtrain = X[train_index,:]
            ztrain = z[train_index]

            Xtest = X[test_index,:]
            ztest = z[test_index]

            Xtrain_scaled,Xtest_scaled,ztrain_scaled,ztest_scaled = centering(Xtrain,Xtest,ztrain,ztest)


            model = Lasso(X,z,lam=lam)
            zpred = model.predict(Xtest_scaled)

            error[i] = MSE(ztest_scaled.ravel(),zpred)
            errorR2[i] = R2(ztest_scaled.ravel(),zpred)
            i += 1


    return np.mean(error[1:-1]),np.mean(errorR2[1:-1])
