from functions import *
from bootstrap import biasVariance



def useCrossValidation(N,z_tuning,maxdegree,k,method,lam=0):

    x,y,z = Data(N,z_tuning)
    z = z.ravel()
    X_maxdegree = create_X(x, y, maxdegree)

    polydegree = np.zeros(maxdegree)
    MSE_test = np.zeros(maxdegree)
    R2_test = np.zeros(maxdegree)


    for degree in range(1,maxdegree+1):
        X = features(X_maxdegree,degree)

        polydegree[degree-1] = degree
        MSE_test[degree-1],R2_test[degree-1] = crossValidation(X,z,k,method,lam)


    return polydegree, MSE_test


def plotKfold(N,z_tuning,maxdegree,method,plot=True,lam=0):
    if plot:
        plt.style.use('seaborn-v0_8')
        for i in range(5,11):
            polydegree,MSE_test_cross = useCrossValidation(N,z_tuning,maxdegree,i,method,lam=lam)
            plt.plot(polydegree,MSE_test_cross,"-o",markersize=5,label=f"k = {i}")

        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.legend()
        plt.xticks(polydegree)
        plt.show()
