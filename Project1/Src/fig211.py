from functions import *


# Make data.
def figHastie(N,z_tuning,maxdegree,plot=True):
    x,y,z = Data(N,z_tuning)
    z = z.ravel()

    MSE_test = np.zeros(maxdegree)
    MSE_train = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)

    X = create_X(x, y, maxdegree)
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)

    print("Calculating error for  degree..")
    for degree in range(1,maxdegree+1):
        print(degree)

        X_train_split,X_test_split = features(X_train,degree),features(X_test,degree)
        X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled = centering(X_train_split,X_test_split,z_train,z_test)

        beta = OLS(X_train_scaled,z_train_scaled)
        zpredict_test = X_test_scaled@beta
        zpredict_train = X_train_scaled@beta

        MSE_train[degree-1] = MSE(z_train_scaled,zpredict_train)
        MSE_test[degree-1]= MSE(z_test_scaled,zpredict_test)
        polydegree[degree-1] = degree

    if plot:
        plt.style.use('seaborn-v0_8')
        plt.plot(polydegree,MSE_train,color="midnightblue",label="Training")
        plt.plot(polydegree,MSE_test,'--',color="brown",label="Test")
        plt.xlabel("Complexity")
        plt.ylabel("MSE")
        plt.xticks(polydegree)
        plt.legend()

        plt.show()
