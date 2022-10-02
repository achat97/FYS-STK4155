from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from functions import *



# Make data.
N = 22
maxdegree = 15
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x,y)
np.random.seed(2018)
sigma2 = 0.1
noise = np.random.normal(0,sigma2,(N,N))
nbootstraps = N


z = FrankeFunction(x,y)+noise
z = z.ravel()

polydegree = np.zeros(maxdegree)
MSE_test = np.zeros(maxdegree)
bias_test = np.zeros(maxdegree)
var_test = np.zeros(maxdegree)

MSE_train = np.zeros(maxdegree)
bias_train = np.zeros(maxdegree)
var_train = np.zeros(maxdegree)

X = create_X(x, y, maxdegree)
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
X_train_scaled_all,X_test_scaled_all,z_train_scaled,z_test_scaled = mean_scaler(X_train,X_test,z_train,z_test)
z_train_scaled,z_test_scaled = z_train_scaled[:,None],z_test_scaled[:,None]



for degree in range(1,maxdegree+1):
    X_train_scaled,X_test_scaled = features(X_train_scaled_all,X_test_scaled_all,degree)

    z_pred_test = bootstrap_OLS(X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled,nbootstraps,"test")
    z_pred_train = bootstrap_OLS(X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled,nbootstraps,"train")


    polydegree[degree-1] = degree
    MSE_test[degree-1] = np.mean( np.mean((z_test_scaled - z_pred_test)**2, axis=1, keepdims=True) )
    bias_test[degree-1] = np.mean( (z_test_scaled - np.mean(z_pred_test, axis=1, keepdims=True))**2 )
    var_test[degree-1] = np.mean( np.var(z_pred_test, axis=1, keepdims=True) )

    MSE_train[degree-1] = np.mean( np.mean((z_train_scaled - z_pred_train)**2, axis=1, keepdims=True) )
    bias_train[degree-1] = np.mean( (z_train_scaled - np.mean(z_pred_train, axis=1, keepdims=True))**2 )
    var_train[degree-1] = np.mean( np.var(z_pred_train, axis=1, keepdims=True) )

    #print('Polynomial degree:', degree)
    #print('Error:', MSE[degree-1])
    #print('Bias^2:', bias[degree-1])
    #print('Var:', var[degree-1])
    #print('{} >= {} + {} = {}'.format(MSE[degree-1], bias[degree-1], var[degree-1], bias[degree-1]+var[degree-1]))
    #print("_______________________________________________________________")




plt.plot(polydegree,MSE_test,"-o",color="r",label=r"$MSE_{test}$")
plt.plot(polydegree,bias_test,"-o",color="b",label=r"$Bias_{test}$")
plt.plot(polydegree,var_test,"-o",color="g",label=r"$Variance_{test}$")
plt.title(f"Test: N={N}, $\sigma^2 = {sigma2}$ ")
plt.legend()
plt.show()


plt.plot(polydegree,MSE_train,"-o",color="r",label=r"$MSE_{training}$")
plt.plot(polydegree,bias_train,"-o",color="b",label=r"$Bias_{training}$")
plt.plot(polydegree,var_train,"-o",color="g",label=r"$Variance_{training}$")
plt.title(f"Training: N={N}, $\sigma^2 = {sigma2}$ ")
plt.legend()
plt.show()
