from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
from FFNN import *
import os


save_directory = '/Users/achrafatifi/Documents/FYS-STK4155/FYS-STK4155/Project2/Src/'

# Navigate one step back to the parent directory
parent_directory = os.path.join(save_directory, '..')

# Navigate into the 'Figures' directory
figures_directory = os.path.join(parent_directory, 'Figures')

warnings.filterwarnings('ignore')

np.random.seed(2014)
cancer = load_breast_cancer()

X = cancer.data
t = cancer.target
t = t.reshape(-1,1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, t_train, t_test = train_test_split(X,t,test_size=0.2)

def heatmap_ln():
    """
    Creates heatmap for accuracy using a neural network for different numbers of hidden layers and neurons
    """

    nlayers = np.arange(0,10)
    nneurons = np.arange(5,55,5)
    accuracy = np.zeros((len(nlayers),len(nneurons)))

    for i in range(len(nlayers)):
        print('i:',i)
        for j in range(len(nneurons)):
            print('j:',j)
            w,b = FFNN(X_train,t_train,nlayers[i],nneurons[j],100,80,LRELU,sigmoid,LRELU_derivative,sigmoid_derivative,crossentropy_derivative)
            a = feed_forward_pass(X_test,w,b,nlayers[i],LRELU,sigmoid)
            accuracy[i][j] = accuracy_score(np.where(a[-1] > 0.5, 1, 0),t_test)
        
    xticks = [f"${neuron}$" for neuron in nneurons]
    yticks = [f"${layer}$" for layer in nlayers]

    fig1, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_ln_accuracy = sns.heatmap(accuracy, annot=True, cmap="YlOrRd_r",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax, vmin = 0, vmax = 1)

    cbar_ln = heatmap_ln_accuracy.collections[0].colorbar
    cbar_ln.ax.tick_params(labelsize=19)  
    cbar_ln.set_label('Accuracy', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_ylabel("L-1",fontsize=19)
    ax.set_xlabel("$N_l$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapClass_ln.png'))

def heatmap_eb():
    """
    Creates one heatmap for accuracy using a neural network
    with 3 hidden layer and 20 neurons, for different numbers of epochs and mini-batches
    """

    nepochs = [10,50,100,150,200,250,300,350,400,450]
    nbatches = [10,20,30,40,50,60,70,80,90,100]
    accuracy = np.zeros((len(nepochs),len(nbatches)))

    i=0
    j=0
    for epochs in nepochs:
        print('i:',i)
        for batches in nbatches:
            print('j:',j)
            w,b = FFNN(X_train,t_train,3,20,epochs,batches,LRELU,sigmoid,LRELU_derivative,sigmoid_derivative,crossentropy_derivative)
            a = feed_forward_pass(X_test,w,b,3,LRELU,sigmoid)
            accuracy[i][j] = accuracy_score(np.where(a[-1] > 0.5, 1, 0),t_test)

            j+=1
        print()
        j=0
        i+=1
        

    xticks = [f"${batch}$" for batch in nbatches]
    yticks = [f"${epoch}$" for epoch in nepochs]

    fig1, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_eb_accuracy = sns.heatmap(accuracy, annot=True, cmap="YlOrRd_r",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax, vmin = 0, vmax = 1)

    cbar_eb = heatmap_eb_accuracy.collections[0].colorbar
    cbar_eb.ax.tick_params(labelsize=19)  
    cbar_eb.set_label('Accuracy', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_title("Accuracy",fontsize=19)
    ax.set_ylabel("Epochs",fontsize=19)
    ax.set_xlabel("Batches",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'HeatmapClass_eb.png'))

def lmbdaplot():
    """
    Plots the accuracy of neural network with 3 hidden layer and 20 neurons for different values of the hyperparameter for L2-regularisation
    """

    lmbda = np.array([0] + [10**(-i) for i in range(9, 0, -1)] + [0.1 * i for i in range(2, 11)])
    accuracy = np.zeros(len(lmbda))
    max = 0
    indx = 0
    for i in range(len(lmbda)):
        w,b = FFNN(X_train,t_train,3,20,10,100,LRELU,sigmoid,LRELU_derivative,sigmoid_derivative,crossentropy_derivative,lmbda=lmbda[i])
        a = feed_forward_pass(X_test,w,b,3,LRELU,sigmoid)
        accuracy[i] = accuracy_score(np.where(a[-1] > 0.5, 1, 0),t_test)

        if accuracy[i]>max:
            indx = i
            max = accuracy[i]
    
    print(f"Largest accuracy for lambda={lmbda[indx]} with an accuracy of {max}")
    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(lmbda,accuracy,label="f(x)",linewidth=9,color='midnightblue')
    ax.set_xlabel("$\lambda$",fontsize=28)
    ax.set_ylabel("Accuracy",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.savefig(os.path.join(figures_directory, 'graph_lmbda_nnClass'))

def activaton_comparison():
    """
    Plots the accuracy as a function for number of epochs for three different activation functions for the hidden layers: Sigmoind, ReLU and LReLU
    """

    epochs = np.arange(0,300,10)
    accuracy_sigmoid = np.zeros(len(epochs))
    accuracy_RELU = np.zeros(len(epochs))
    accuracy_LRELU = np.zeros(len(epochs))
    
    i = 0
    for epoch in epochs: 
        print("epoch",epoch)
        w_sig,b_sig = FFNN(X_train,t_train,3,2,epoch,100,sigmoid,sigmoid,sigmoid_derivative,sigmoid_derivative,crossentropy_derivative)
        w_rel,b_rel = FFNN(X_train,t_train,3,20,epoch,100,RELU,sigmoid,RELU_derivative,sigmoid_derivative,crossentropy_derivative)
        w_lrel,b_lrel = FFNN(X_train,t_train,3,20,epoch,100,LRELU,sigmoid,LRELU_derivative,sigmoid_derivative,crossentropy_derivative)

        a_sig = feed_forward_pass(X_test,w_sig,b_sig,3,sigmoid,sigmoid)
        a_rel = feed_forward_pass(X_test,w_rel,b_rel,3,RELU,sigmoid)
        a_lrel = feed_forward_pass(X_test,w_lrel,b_lrel,3,LRELU,sigmoid)

        accuracy_sigmoid[i] = accuracy_score(np.where(a_sig[-1] > 0.5, 1, 0),t_test)
        accuracy_RELU[i] = accuracy_score(np.where(a_rel[-1] > 0.5, 1, 0),t_test)
        accuracy_LRELU[i] = accuracy_score(np.where(a_lrel[-1] > 0.5, 1, 0),t_test)

        i+=1


    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))

    ax.plot(epochs,accuracy_sigmoid,label="Sigmoid",linewidth=9,color='midnightblue')
    ax.plot(epochs,accuracy_RELU,label="RELU",linewidth=9,color='red')
    ax.plot(epochs,accuracy_LRELU,label="LRELU",linewidth=9,color='green')
    ax.set_xlabel("Epochs",fontsize=28)
    ax.set_ylabel("Accuracy",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)
    plt.savefig(os.path.join(figures_directory, 'graphs_nn_class_activation.png'))

def lmbdaplot_logistic():
    """
    Plots the accuracy of logistic regression for different values of the hyperparameter for L2-regularisation
    """

    lmbda = np.array([0] + [10**(-i) for i in range(9, 0, -1)] + [0.1 * i for i in range(2, 11)])
    accuracy = np.zeros(len(lmbda))
    max = 0
    indx = 0
    for i in range(len(lmbda)):
        w,b = FFNN(X_train,t_train,0,0,10,100,identity,sigmoid,identity_derivative,sigmoid_derivative,crossentropy_derivative,lmbda=lmbda[i])
        a = feed_forward_pass(X_test,w,b,0,identity,sigmoid)
        accuracy[i] = accuracy_score(np.where(a[-1] > 0.5, 1, 0),t_test)

        if accuracy[i]>max:
            indx = i
            max = accuracy[i]
    
    print(f"Largest accuracy for lambda={lmbda[indx]} with an accuracy of {max}")
    plt.style.use('seaborn-v0_8')
    fig1, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(lmbda,accuracy,label="f(x)",linewidth=9,color='midnightblue')
    ax.set_xlabel("$\lambda$",fontsize=28)
    ax.set_ylabel("Accuracy",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.savefig(os.path.join(figures_directory, 'graph_lmbda_logistic'))

def sklLogistic():
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X_train, t_train)
    print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,t_test)))

#heatmap_ln()
#heatmap_eb()
#lmbdaplot()
#activaton_comparison()
#lmbdaplot_logistic()
#sklLogistic()