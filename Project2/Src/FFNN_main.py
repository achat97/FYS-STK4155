import pandas_flavor
from FFNN import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

#Remove all the these paths to save figures where its run or create own path
save_directory = '/Users/achrafatifi/Documents/FYS-STK4155/FYS-STK4155/Project2/Src/'

# Navigate one step back to the parent directory
parent_directory = os.path.join(save_directory, '..')

# Navigate into the 'Figures' directory
figures_directory = os.path.join(parent_directory, 'Figures')


np.random.seed(2014)

n = 400
x = np.random.rand(n,1)
y = 2+3*x+4*x**2+0.1*np.random.randn(n,1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    

def heatmap_ln():
    """
    Creates two heatmaps, one for MSE and one for R2, using a neural network for different numbers of hidden layers and neurons
    """

    nlayers = np.arange(0,10)
    nneurons = np.arange(5,55,5)
    mse = np.zeros((len(nlayers),len(nneurons)))
    r2 = np.zeros((len(nlayers),len(nneurons)))

    for i in range(len(nlayers)):
        print('i:',i)
        for j in range(len(nneurons)):
            print('j:',j)
            w,b = FFNN(x_train,y_train,nlayers[i],nneurons[j],300,80,sigmoid,identity,sigmoid_derivative,identity_derivative,mse_derivative)
            a = feed_forward_pass(x_test,w,b,nlayers[i],sigmoid,identity)
            mse[i][j] = MSE(y_test, a[-1])
            r2[i][j] = R2(y_test, a[-1])
        

    xticks = [f"${neuron}$" for neuron in nneurons]
    yticks = [f"${layer}$" for layer in nlayers]

    fig1, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_ln_mse = sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax)

    cbar_ln = heatmap_ln_mse.collections[0].colorbar
    cbar_ln.ax.tick_params(labelsize=19)  
    cbar_ln.set_label('MSE', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_ylabel("L-1",fontsize=19)
    ax.set_xlabel("$N_l$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapReg_ln_mse.png'))

    fig2, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_ln_r2 = sns.heatmap(r2, annot=True, cmap="YlOrRd_r",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax)

    cbar_ln = heatmap_ln_r2.collections[0].colorbar
    cbar_ln.ax.tick_params(labelsize=19)  
    cbar_ln.set_label('$R^2$', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_ylabel("L-1",fontsize=19)
    ax.set_xlabel("$N_l$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapReg_ln_r2.png'))

def heatmap_eb():
    """
    Creates two heatmaps, one for MSE and one for R2 using a neural network
    with 1 hidden layer and 35 neurons, for different numbers of epochs and mini-batches
    """

    nepochs = [10,50,100,150,200,250,300,350,400,450]
    nbatches = [2,5,10,20,30,40,50,60,70,80]
    mse = np.zeros((len(nepochs),len(nbatches)))
    r2 = np.zeros((len(nepochs),len(nbatches)))

    i=0
    j=0
    for epochs in nepochs:

        print('i:',i)
        for batches in nbatches:
            print('j:',j)
            w,b = FFNN(x_train,y_train,1,35,epochs,batches,sigmoid,identity,sigmoid_derivative,identity_derivative,mse_derivative)
            a = feed_forward_pass(x_test,w,b,1,sigmoid,identity)
            mse[i][j] = MSE(y_test, a[-1])
            r2[i][j] = R2(y_test, a[-1])

            j+=1
        print()
        j=0
        i+=1
        

    xticks = [f"${batch}$" for batch in nbatches]
    yticks = [f"${epoch}$" for epoch in nepochs]

    fig1, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_eb_mse = sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax)

    cbar_eb = heatmap_eb_mse.collections[0].colorbar
    cbar_eb.ax.tick_params(labelsize=19)  
    cbar_eb.set_label('MSE', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_ylabel("Epochs",fontsize=19)
    ax.set_xlabel("Batches",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapReg_eb_mse.png'))

    fig2, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_eb_r2 = sns.heatmap(r2, annot=True, cmap="YlOrRd_r",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax)

    cbar_eb = heatmap_eb_r2.collections[0].colorbar
    cbar_eb.ax.tick_params(labelsize=19)  
    cbar_eb.set_label('$R^2$', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_xlabel("Batches",fontsize=19)
    ax.set_ylabel("Epochs",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapReg_eb_r2.png'))

def lmbdaplot():
    """
    Plots the MSE of neural network with 1 hidden layer and 35 neurons for different values of the hyperparameter for L2-regularisation
    """

    lmbda = np.array([0] + [10**(-i) for i in range(9, 0, -1)] + [0.1 * i for i in range(2, 11)])
    mse = np.zeros(len(lmbda))
    for i in range(len(lmbda)):
        w,b = FFNN(x_train,y_train,1,35,300,80,sigmoid,identity,sigmoid_derivative,identity_derivative,mse_derivative,lmbda=lmbda[i])
        a = feed_forward_pass(x_test,w,b,1,sigmoid,identity)
        mse[i] = MSE(y_test, a[-1])

    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(lmbda,mse,label="f(x)",linewidth=9,color='midnightblue')
    ax.set_xlabel("$\lambda$",fontsize=28)
    ax.set_ylabel("MSE",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.savefig(os.path.join(figures_directory, 'graph_lmbda_nn'))

def plotGraphs():
    """
    Plots predictions of OLS, Ridge, own NN and skl NN as well as the true function
    """

    X = Design(x,2)

    xn = np.linspace(0,1,1000)
    yn = 2+3*xn+4*xn**2

    y_OLS = X@OLS(X,y)
    y_ridge = X@Ridge(X,y,1e-5)

    w,b = FFNN(x_train,y_train,1,35,300,80,sigmoid,identity,sigmoid_derivative,identity_derivative,mse_derivative)
    a = feed_forward_pass(x,w,b,1,sigmoid,identity)

    dnn = MLPRegressor(hidden_layer_sizes=(35), activation='logistic',alpha=0, max_iter=5000,solver='adam',batch_size=4)
    dnn.fit(x_train, y_train.ravel())
    askl_test = dnn.predict(x_test)
    askl = dnn.predict(x)
    print('MSE - skl:',MSE(y_test,askl_test.reshape(-1,1)))

    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))

    ax.plot(xn,yn,label="f(x)",linewidth=9,color='midnightblue')
    ax.scatter(x,y_OLS,label="OLS",s=200,color='red')
    ax.scatter(x,y_ridge,label="Ridge",s=200,color="green")
    ax.scatter(x,a[-1],label="NN",s=200,color='orange')
    ax.scatter(x,askl.reshape(-1,1),label="NN - skl ",s=200,color='purple')
    ax.set_xlabel("x",fontsize=28)
    ax.set_ylabel("y",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)

    plt.savefig(os.path.join(figures_directory, 'graphs_nn_regression'))

def activaton_comparison():
    """
    Plots the MSE as a function for number of epochs for three different activation functions for the hidden layers: Sigmoind, ReLU and LReLU
    """

    epochs = np.arange(0,300,10)
    mse_sigmoid = np.zeros(len(epochs))
    mse_RELU = np.zeros(len(epochs))
    mse_LRELU = np.zeros(len(epochs))
    
    i = 0
    for epoch in epochs: 
        print("epoch",epoch)
        w_sig,b_sig =FFNN(x_train,y_train,1,35,epoch,80,sigmoid,identity,sigmoid_derivative,identity_derivative,mse_derivative)
        w_rel,b_rel =FFNN(x_train,y_train,1,35,epoch,80,RELU,identity,RELU_derivative,identity_derivative,mse_derivative)
        w_lrel,b_lrel =FFNN(x_train,y_train,1,35,epoch,80,LRELU,identity,LRELU_derivative,identity_derivative,mse_derivative)

        a_sig = feed_forward_pass(x_test,w_sig,b_sig,1,sigmoid,identity)
        a_rel = feed_forward_pass(x_test,w_rel,b_rel,1,RELU,identity)
        a_lrel= feed_forward_pass(x_test,w_lrel,b_lrel,1,LRELU,identity)

        mse_sigmoid[i] = MSE(y_test,a_sig[-1])
        mse_RELU[i] = MSE(y_test,a_rel[-1])
        mse_LRELU[i] = MSE(y_test,a_lrel[-1])

        i+=1


    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))

    ax.plot(epochs,mse_sigmoid,label="Sigmoid",linewidth=9,color='midnightblue')
    ax.plot(epochs,mse_RELU,label="RELU",linewidth=9,color='red')
    ax.plot(epochs,mse_LRELU,label="LRELU",linewidth=9,color='green')
    ax.set_xlabel("Epochs",fontsize=28)
    ax.set_ylabel("MSE",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)
    plt.savefig(os.path.join(figures_directory, 'graphs_nn_regression_activation.png'))


#heatmap_ln()
#heatmap_eb()
#lmbdaplot()
#plotGraphs()
#activaton_comparison()