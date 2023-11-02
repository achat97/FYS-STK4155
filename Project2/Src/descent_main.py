from functions import *
from GD_analytical import *
from SGD_analytical import *
import seaborn as sns

np.random.seed(2014)


n = 400
x = np.random.rand(n,1)
y = 2+3*x+4*x**2+0.1*np.random.randn(n,1)

p = 2
X = Design(x,p)
beta = np.random.randn(p+1,1)

lr = np.logspace(-10,-1,10)
lmbda = np.insert(np.logspace(-9,0,10),0,0)



def plotHeatmapGDanalytical():
    mse = np.zeros((len(lr),len(lmbda)))
    min = 100
    for i in range(len(lr)):
        for j in range(len(lmbda)):
            beta_gd = GD_analytical(X,y,beta,2000,lr[i],lmbda[j])
            print(f"lr={lr[i]}, lmbda={lmbda[j]}")
            print(beta_gd)
            print()
            y_gd = X@beta_gd
            mse[i][j] = MSE(y,y_gd)
            print(mse[i][j])
            if mse[i][j]<min:
                min = mse[i][j] = MSE(y,y_gd)
                minlr = lr[i]
                minlmbda = lmbda[j]

    print(f"Smallest MSE={min} with learning rate {minlr} and hyperparameter {minlmbda}")


    # Set the y-axis labels to powers of ten
    yticks = [f"$10^{{{int(np.log10(val))}}}$" for val in lr]

    # Set the x-axis labels to powers of ten
    xticks = [f"$0$"]+[f"$10^{{{int(np.log10(val))}}}$" for val in np.logspace(-9,0,10)]

    sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, cbar_kws={'label': 'MSE'})

    plt.tick_params(left=False, bottom=False)
    plt.title("Accuracy")
    plt.ylabel("$\eta$")
    plt.xlabel("$\lambda$")
    plt.show()



def plotHeatmapSGDanalytical():
    mse = np.zeros((len(lr),len(lmbda)))
    min = 100
    for i in range(len(lr)):
        for j in range(len(lmbda)):
            beta_gd = SGD_analytical(X,y,beta,25,40,lr[i],lmbda[j],gamma=0)
            print(f"lr={lr[i]}, lmbda={lmbda[j]}")
            print(beta_gd)
            print()
            y_gd = X@beta_gd
            mse[i][j] = MSE(y,y_gd)
            if mse[i][j]<min:
                min = mse[i][j] = MSE(y,y_gd)
                minlr = lr[i]
                minlmbda = lmbda[j]

    print(f"Smallest MSE={min} with learning rate {minlr} and hyperparameter {minlmbda}")


    # Set the y-axis labels to powers of ten
    yticks = [f"$10^{{{int(np.log10(val))}}}$" for val in lr]

    # Set the x-axis labels to powers of ten
    xticks = [f"$0$"]+[f"$10^{{{int(np.log10(val))}}}$" for val in np.logspace(-9,0,10)]

    sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks,cbar_kws={'label': 'MSE'})

    plt.tick_params(left=False, bottom=False)
    plt.title("Accuracy")
    plt.ylabel("$\eta$")
    plt.xlabel("$\lambda$")
    plt.show()


def plothHeatmapSGDepochsbatches():
    min = 100
    nepochs = [10,20,30,40,50,60,70,80,90,100]
    nbatches = [10,40,80,120,160,200,240,280,320,360]

    mse = np.zeros((len(nepochs),len(nbatches)))


    i=0
    j=0
    for epochs in nepochs:
        for batches in nbatches:
            beta_gd = SGD_analytical(X,y,beta,epochs,batches,0.1,0,gamma=0)
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

    yticks = [f"${epoch}$" for epoch in nepochs]

    # Set the x-axis labels to powers of ten
    xticks = [f"${batch}$" for batch in nbatches]

    sns.heatmap(mse, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks,cbar_kws={'label': 'MSE'})

    plt.tick_params(left=False, bottom=False)
    plt.title("Accuracy")
    plt.ylabel("Epochs")
    plt.xlabel("Batches")
    plt.show()


def momentumGDanalytical():
    momentum = np.linspace(0,1,11)

    for i in range(len(momentum)):
        beta_mom = GD_analytical(X,y,beta,2000,0.1,1e-5,momentum[i])
        mse_mom = MSE(y,X@beta_mom)
        
        print(f"momentum = {momentum[i]:.1f}")
        print(f"MSE = {mse_mom}")
        print()


def momentumSGDanalytical():
    momentum = np.linspace(0,1,11)

    for i in range(len(momentum)):
        beta_mom = SGD_analytical(X,y,beta,20,80,0.1,gamma=momentum[i])
        mse_mom = MSE(y,X@beta_mom)
        
        print(f"momentum = {momentum[i]:.1f}")
        print(f"MSE = {mse_mom}")
        print()

#print(MSE(y,X@SGD_analytical_scheduler(X,y,beta,1000,390,1,20)))

#print(MSE(y,X@GD_analytical_Adagrad(X,y,beta,2000,1,gamma=0.9)))

#print(MSE(y,X@SGD_analytical_Adagrad(X,y,beta,20,80,0.9,gamma=0.9)))

#print(MSE(y,X@GD_analytical_RMSprop(X,y,beta,5000,0.01,rho=0.9,delta=1e-6,lmbda=0,gamma=0)))

#print(MSE(y,X@SGD_analytical_RMSprop(X,y,beta,20,80,0.01,0.9)))

#print(MSE(y,X@GD_analytical_ADAM(X,y,beta,2000,eta=5,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0))) 

#print(MSE(y,X@SGD_analytical_ADAM(X,y,beta,20,80,eta=0.9,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0))) 

