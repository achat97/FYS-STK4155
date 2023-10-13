from functions import *
from erroranalysis import *
from bootstrap import *
from crossvalidation import *
from fig211 import *



#plots values of beta-parameters up to degree 5
#plots beta-parameters of degree 5 with an errorbar of 1 std
#plots MSE_train vs MSE_test as well as R^2-score using OLS up to degree 5
polydegree,MSE_test,R2_test = error_analysis(20,0.1,5,"OLS")

#plots MSE_test vs MSE_train up to degree 14
figHastie(20,0.1,14)

#plots bias-variance tradeoff using OLS using two different amounts of datapoints
polydegree, MSE_test_boot = biasVariance(20,0.1,14,20**2,"OLS")
polydegree, MSE_test_boot = biasVariance(50,0.1,14,50**2,"OLS")

#plots MSE_test for different numbers of k-folds(from 5 to 10) using OLS
plotKfold(20,0.1,14,"OLS")


def splitVbootVcross(N,z_tuning,maxdegree,B,k,method,lam=0):
    polydegree, MSE_test_boot = biasVariance(N,z_tuning,maxdegree,B,method,plot=False,lam=lam)
    polydegree,MSE_test_cross = useCrossValidation(N,z_tuning,maxdegree,k,method,lam=lam)
    polydegree,MSE_test,R2 = error_analysis(N,z_tuning,maxdegree,method,plot=False,lam=lam)

    plt.style.use('seaborn-v0_8')
    plt.plot(polydegree,MSE_test_cross,"-o",color="brown",markersize=5,label=f"CV, folds = {k}")
    plt.plot(polydegree,MSE_test_boot,"-s",color="midnightblue",markersize=5,label=f"Bootstrap, b = {B} ")
    plt.plot(polydegree,MSE_test,"-^",color="darkgoldenrod",markersize=5,label="Regular split")

    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.legend()
    plt.xticks(polydegree)
    plt.show()
    return MSE_test

#plots MSE using OLS with bootstrap, CV and no resampling for different amounts of datapoints

MSE_test=splitVbootVcross(20,0.1,14,20**2,10,"OLS")
print("Error, OLS:", MSE_test)

MSE_test=splitVbootVcross(80,0.1,14,20**2,10,"OLS")
print("Error, OLS:", MSE_test)


#plots MSE_test for different numbers of k-folds(from 5 to 10) using Ridge
plotKfold(20,0.1,14,"Ridge",lam=0.001)



#This is not used in the report but useful function which plots MSE for several hyperparameters
def CVLamb(N,z_tuning,maxdegree,k,method,lamb):

    i=0
    for lam in lamb:
        polydegree,MSE_test_cross = useCrossValidation(N,z_tuning,maxdegree,k,method,lam=lam)
        plt.style.use('seaborn-v0_8')
        plt.plot(polydegree,MSE_test_cross,"-o",color=colors[i],markersize=5,label=f"$\lambda$={lam}")
        i += 1

    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.legend()
    plt.xticks(polydegree)
    plt.legend()
    plt.show()


#plots bias-variance tradeoff using ridge regression for two different amounts of datapoints and hyperparameters
polydegree, MSE_test_boot = biasVariance(20,0.1,14,20**2,"Ridge",lam=0.1)
polydegree, MSE_test_boot2 = biasVariance(20,0.1,14,20**2,"Ridge",lam=10**(-9))
polydegree, MSE_test_boot2 = biasVariance(50,0.1,14,50**2,"Ridge",lam=10**(-9))

#plots MSE using ridgeregression with bootstrap, CV and no resampling for two different amounts of datapoints and hyperparameters


MSE_test = splitVbootVcross(20,0.1,14,20**2,10,"Ridge",lam=10**(-9))
print("Error, Ridge:", MSE_test)



#plots a heatmap of MSE as a function of hyperparameter and polynomial degree
def Heatmaplamb(N,z_tuning,maxdegree,k,method,lamb):
    MSE_test = []

    i = 0
    for lam in lamb:
        polydegree,MSE_test_cross = useCrossValidation(N,z_tuning,maxdegree,k,method,lam=lam)
        MSE_test.append(MSE_test_cross.tolist())
        i += 1

    MSE_test = np.array(MSE_test)
    polydegree, lamb = np.meshgrid(polydegree,lamb)

    nlevels = np.linspace(np.min(MSE_test), np.max(MSE_test),60)

    index_min = np.unravel_index(MSE_test.argmin(), MSE_test.shape)
    minMSE = MSE_test[index_min[0]][index_min[1]]
    optDeg = polydegree[index_min[0]][index_min[1]]
    optLam = lamb[index_min[0]][index_min[1]]

    print("Error: ",minMSE)
    print("Optimal degree: ",optDeg)
    print("Optimal hyperparameter: ", optLam)

    plt.yscale("log")
    heatmap = plt.contourf(polydegree,lamb,MSE_test,levels=nlevels,cmap="RdYlBu_r")
    plt.scatter(optDeg,optLam,marker="x",s=100,color="black")
    cb = plt.colorbar(heatmap,format="{x:.3f}")
    cb.ax.set_ylabel('MSE', rotation=270,labelpad=15)
    cb.ax.locator_params(nbins=6)
    plt.xlabel("Degree")
    plt.ylabel(r"$\lambda$")
    plt.show()

    return MSE_test,polydegree,lamb

#plots a heatmap of the MSE (ridge and lasso) with degree on the x-axis and hyperparameters on the y-axis. Also prints most optimal hyperparameter and degree.
lamb = np.logspace(-1,-14,14)
MSEheat,polydegree,lmb = Heatmaplamb(20,0.1,14,10,"Ridge",lamb)
MSEheat,polydegree,lmb = Heatmaplamb(20,0.1,14,9,"Lasso",lamb)


#Same as the plots for ridge, just for lasso and with slightly different k-fold and hyperparameter
plotKfold(20,0.1,14,"Lasso",lam=10**(-6))


polydegree, MSE_test_boot = biasVariance(20,0.1,14,20**2,"Lasso",lam=0.1)
polydegree, MSE_test_boot2 = biasVariance(20,0.1,14,20**2,"Lasso",lam=10**(-6))


MSE_test=splitVbootVcross(20,0.1,14,20**2,9,"Lasso",lam=10**(-14))
print("Error, Lasso:", MSE_test)
