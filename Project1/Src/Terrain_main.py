from functions import *

def crossValidation_terrain(maxdegree,k,method,lam=0):
    terrain,x,y,z = Data_terrain("SRTM_data_Morocco.tif",30,600,2600)
    z = z.ravel()
    X_maxdegree = create_X(x, y, maxdegree)

    polydegree = np.zeros(maxdegree)
    MSE_test = np.zeros(maxdegree)
    R2_test = np.zeros(maxdegree)


    for degree in range(1,maxdegree+1):
        X = features(X_maxdegree,degree)

        polydegree[degree-1] = degree
        MSE_test[degree-1], R2_test[degree-1] = crossValidation(X,z,k,method,lam)


    return polydegree, MSE_test, R2_test


def plotKfold_terrain(maxdegree,method,plot=True,lam=0):
    if plot:
        plt.style.use('seaborn-v0_8')
        for i in range(5,11):
            polydegree,MSE_test_cross,R2_test_cross = crossValidation_terrain(maxdegree,i,method,lam=lam)
            plt.plot(polydegree,R2_test_cross,"-o",markersize=5,label=f"k = {i}")
            print("k=",i)
            print("Max R2",np.max(R2_test_cross))
            print("Degree=",np.argmax(R2_test_cross)+1)
            print("----------------------------")

        plt.xlabel("Complexity")
        plt.ylabel(r"$R^2-Score$")
        plt.legend()
        plt.xticks(polydegree)
        plt.show()

#plot R2-score for different folds
plotKfold_terrain(30,"OLS")
plotKfold_terrain(30,"Ridge",lam=10**(-9))

plotKfold_terrain(30,"Lasso",lam=10**(-9))





def Heatmaplamb_terrain(maxdegree,k,method,lamb):
    MSE_test = []

    i = 0
    for lam in lamb:
        polydegree,MSE_test_cross,R2_test_cross = crossValidation_terrain(maxdegree,k,method,lam=lam)
        MSE_test.append(MSE_test_cross.tolist())
        i += 1

    MSE_test = np.array(MSE_test)
    polydegree, lamb = np.meshgrid(polydegree,lamb)

    nlevels = np.linspace(np.min(MSE_test),np.max(MSE_test),60)

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
lamb = np.logspace(-4,-14,10)
MSEheat,polydegree,lmb = Heatmaplamb_terrain(30,10,"Ridge",lamb)
MSEheat,polydegree,lmb = Heatmaplamb_terrain(30,10,"Lasso",lamb)
