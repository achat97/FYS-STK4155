from functions import *
from GD_analytical import *
from GD_autodiff import *
from SGD_analytical import *
from SGD_autodiff import *
import seaborn as sns
from jax import grad
import os

#Remove all the these paths to save figures where its run or create own path
save_directory = '/Users/achrafatifi/Documents/FYS-STK4155/FYS-STK4155/Project2/Src/' 

# Navigate one step back to the parent directory
parent_directory = os.path.join(save_directory, '..')

# Navigate into the 'Figures' directory
figures_directory = os.path.join(parent_directory, 'Figures')


np.random.seed(2014)

n = 400 #number of data points
x = np.random.rand(n,1) #input
y = 2+3*x+4*x**2+0.1*np.random.randn(n,1) #output

p = 2 #polynomial degree 

X = Design(x,p) #design matrix
beta = np.random.randn(p+1,1) #initialise parameters

lr = np.logspace(-10,-1,10) #learning rates
lmbda = np.insert(np.logspace(-9,0,10),0,0)#hyperparameters

"""
WARNING! 

These methods use automatic differentiation, so they may take a substantial amount of time.
You can use an analytical gradient instead by adding "_analytical" after GD or SGD.

Examples: GD->GD_analytical, SGD_RMSProp->SGD_analytical_RMSprop

Also, you have to uncomment the function calls at the end of the script to generate the results shown in report
"""


def plotHeatmapGD():
    """
    Creates two heatmaps, one for MSE and one for R2, using plain gradient descent for different learning rates and hyperparameters.
    """

    mse = np.zeros((len(lr),len(lmbda)))
    r2 = np.zeros((len(lr),len(lmbda)))
    min = 100
    for i in range(len(lr)):
        for j in range(len(lmbda)):
            beta_gd = GD(X,y,beta,2000,eta=lr[i],lmbda=lmbda[j]) 
            print(f"lr={lr[i]}, lmbda={lmbda[j]}")
            print(beta_gd)
            y_gd = X@beta_gd
            mse[i][j] = MSE(y,y_gd)
            r2[i][j] = R2(y,y_gd)
            print("MSE=",mse[i][j])
            print()
            if mse[i][j]<min:
                min = mse[i][j] = MSE(y,y_gd)
                minlr = lr[i]
                minlmbda = lmbda[j]

    print(f"Smallest MSE={min} with learning rate {minlr} and hyperparameter {minlmbda}")

    xticks = [f"$0$"]+[f"$10^{{{int(np.log10(val))}}}$" for val in np.logspace(-9,0,10)]
    yticks = [f"$10^{{{int(np.log10(val))}}}$" for val in lr]

    fig1, ax = plt.subplots(layout='constrained', figsize=(13,11))
    heatmap_mse = sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, vmin = 0 ,annot_kws={"fontsize":22} ,ax=ax)

    cbar_mse = heatmap_mse.collections[0].colorbar
    cbar_mse.ax.tick_params(labelsize=19)  # Adjust font size of colorbar ticks
    cbar_mse.set_label('MSE', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy")
    ax.set_ylabel("$\eta$",fontsize=19)
    ax.set_xlabel("$\lambda$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapGD_MSE.png'))
    

    fig2, ax = plt.subplots(layout='constrained', figsize=(13,11))
    heatmap_r2 = sns.heatmap(r2, annot=True, cmap="YlOrRd_r",yticklabels=yticks, xticklabels=xticks, vmax = 1 ,annot_kws={"fontsize":22}, ax=ax)

    cbar_r2 = heatmap_r2.collections[0].colorbar
    cbar_r2.ax.tick_params(labelsize=19)  # Adjust font size of colorbar ticks
    cbar_r2.set_label("$R^2$", fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy")
    ax.set_ylabel("$\eta$",fontsize=19)
    ax.set_xlabel("$\lambda$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapGD_R2.png'))


def plotHeatmapSGD():
    """
    Creates two heatmaps, one for MSE and one for R2, using stochastic gradient descent for different learning rates and hyperparameters.
    """

    mse = np.zeros((len(lr),len(lmbda)))
    r2 = np.zeros((len(lr),len(lmbda)))

    min = 100
    for i in range(len(lr)):
        for j in range(len(lmbda)):
            beta_gd = SGD(X,y,beta,25,40,lr[i],lmbda[j],gamma=0)
            print(f"lr={lr[i]}, lmbda={lmbda[j]}")
            print(beta_gd)
            print()
            y_gd = X@beta_gd
            mse[i][j] = MSE(y,y_gd)
            r2[i][j] = R2(y,y_gd)
            if mse[i][j]<min:
                min = mse[i][j] = MSE(y,y_gd)
                minlr = lr[i]
                minlmbda = lmbda[j]

    print(f"Smallest MSE={min} with learning rate {minlr} and hyperparameter {minlmbda}")

    xticks = [f"$0$"]+[f"$10^{{{int(np.log10(val))}}}$" for val in np.logspace(-9,0,10)]
    yticks = [f"$10^{{{int(np.log10(val))}}}$" for val in lr]

    fig1, ax = plt.subplots(layout='constrained', figsize=(13,11))
    heatmap_mse = sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, vmin = 0 ,annot_kws={"fontsize":22} ,ax=ax)

    cbar_mse = heatmap_mse.collections[0].colorbar
    cbar_mse.ax.tick_params(labelsize=19)  # Adjust font size of colorbar ticks
    cbar_mse.set_label('MSE', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy")
    ax.set_ylabel("$\eta$",fontsize=19)
    ax.set_xlabel("$\lambda$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapSGD_MSE.png'))
    

    fig2, ax = plt.subplots(layout='constrained', figsize=(13,11))
    heatmap_r2 = sns.heatmap(r2, annot=True, cmap="YlOrRd_r",yticklabels=yticks, xticklabels=xticks, vmax = 1 ,annot_kws={"fontsize":22}, ax=ax)

    cbar_r2 = heatmap_r2.collections[0].colorbar
    cbar_r2.ax.tick_params(labelsize=19)  # Adjust font size of colorbar ticks
    cbar_r2.set_label("$R^2$", fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy")
    ax.set_ylabel("$\eta$",fontsize=19)
    ax.set_xlabel("$\lambda$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapSGD_R2.png'))


def plothHeatmapSGDepochsbatches():
    """
    Creates heatmap of MSE using stochastic gradient descent for different number of epochs and nbatches for learning rate = 0.1
    """

    min = 100
    nepochs = [10,20,30,40,50,60,70,80,90,100]
    nbatches = [10,40,80,120,160,200,240,280,320,360]

    mse = np.zeros((len(nepochs),len(nbatches)))


    i=0
    j=0
    for epochs in nepochs:
        for batches in nbatches:
            beta_gd = SGD_analytical(X,y,beta,epochs,batches,0.1,0,gamma=0) #This one is changed to analytical gradient due to time usage
            print(f"nepochs={epochs}, nbatches={batches}")
            print(beta_gd)
            print()
            y_gd = X@beta_gd
            mse[i][j] = MSE(y,y_gd)
            if mse[i][j]<min:
                min = mse[i][j] = MSE(y,y_gd)
                minepoch = epochs
                minbatch = batches
            
            j+=1
        j=0
        i+=1

    print(f"Smallest MSE={min} after {minepoch} epochs and {minbatch} batches")

    xticks = [f"${batch}$" for batch in nbatches]
    yticks = [f"${epoch}$" for epoch in nepochs]
    
    fig, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_eb = sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax)

    cbar_eb = heatmap_eb.collections[0].colorbar
    cbar_eb.ax.tick_params(labelsize=19)  # Adjust font size of colorbar ticks
    cbar_eb.set_label('MSE', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_ylabel("Epochs",fontsize=19)
    ax.set_xlabel("Batches",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapSGD_EB.png'))


def momentumGD():
    """
    Calculates MSE for different values of the momentum parameter using plain GD for learning rate = 0.1
    """

    momentum = np.linspace(0,1,11)

    for i in range(len(momentum)):
        beta_mom = GD(X,y,beta,2000,0.1,0,momentum[i])
        mse_mom = MSE(y,X@beta_mom)
        
        print(f"momentum = {momentum[i]:.1f}")
        print(f"MSE = {mse_mom}")
        print()


def momentumSGD():
    """
    Calculates MSE for different values of the momentum parameter using SGD with constant learning rate = 0.1
    """

    momentum = np.linspace(0,1,11)

    for j in range(len(momentum)):
        beta_mom = SGD(X,y,beta,20,80,0.1,gamma=momentum[j])
        mse_mom = MSE(y,X@beta_mom)
            
        print(f"momentum = {momentum[j]:.1f}")
        print(f"MSE = {mse_mom}")
        print()


def plotGraphs():
    """
    Plots predictions of OLS, Ridge, GD and SGD as well as the true function
    """

    xn = np.linspace(0,1,1000)
    yn = 2+3*xn+4*xn**2

    y_OLS = X@OLS(X,y)
    y_ridge = X@Ridge(X,y,1e-5)
    y_gd = X@GD_ADAM(X,y,beta,2000,eta=5)
    y_sgd = X@SGD(X,y,beta,20,80,eta=0.1,gamma=0)

    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))

    ax.plot(xn,yn,label="f(x)",linewidth=9,color='midnightblue')
    ax.scatter(x,y_OLS,label="OLS",s=200,color='red')
    ax.scatter(x,y_ridge,label="Ridge",s=200,color="green")
    ax.scatter(x,y_gd,label="GD - Adam",s=200,color='orange')
    ax.scatter(x,y_sgd,label="SGD - constant",s=200,color='purple')
    ax.set_xlabel("x",fontsize=28)
    ax.set_ylabel("y",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)

    plt.savefig(os.path.join(figures_directory, 'graphs_regression.png'))


#plotHeatmapGD()
#momentumGD()

#print(MSE(y,X@GD_Adagrad(X,y,beta,2000,eta=5,gamma=0)))
#print(MSE(y,X@GD_Adagrad(X,y,beta,2000,eta=5,gamma=0.9)))

#print(MSE(y,X@GD_RMSprop(X,y,beta,1500,eta=0.01,gamma=0)))
#print(MSE(y,X@GD_RMSprop(X,y,beta,2000,eta=0.01,gamma=0.8)))

#print(MSE(y,X@GD_ADAM(X,y,beta,2000,eta=5))) 


#plotHeatmapSGD()
#plothHeatmapSGDepochsbatches()
#momentumSGD()

#print(MSE(y,X@SGD_Adagrad(X,y,beta,20,80,eta=1,gamma=0)))
#print(MSE(y,X@SGD_Adagrad(X,y,beta,20,80,eta=1,gamma=0.7)))

#print(MSE(y,X@SGD_RMSprop(X,y,beta,20,80,eta=0.01)))
#print(MSE(y,X@SGD_RMSprop(X,y,beta,20,80,eta=0.001,gamma=0.7)))

#print(MSE(y,X@SGD_ADAM(X,y,beta,20,80,eta=0.1)))

#print(R2(y,X@OLS(X,y)))
#print(R2(y,X@Ridge(X,y,1e-5)))
#print(MSE(y,X@OLS(X,y)))
#print(MSE(y,X@Ridge(X,y,1e-5)))

#plotGraphs()