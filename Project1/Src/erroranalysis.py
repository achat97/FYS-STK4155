from functions import *



def error_analysis(N,z_tuning,maxdegree,method,plot=True,lam=0):

    x,y,z = Data(N,z_tuning)
    z = z.ravel()


    MSE_test = np.zeros(maxdegree)
    R2_test = np.zeros(maxdegree)
    beta_parameters = np.zeros(maxdegree,dtype=object)
    polydegree = np.zeros(maxdegree)

    X = create_X(x, y, maxdegree)
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)

    print("Calculating error for  degree..")
    for degree in range(1,maxdegree+1):
        print(degree)

        X_train_split,X_test_split = features(X_train,degree),features(X_test,degree)
        X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled = centering(X_train_split,X_test_split,z_train,z_test)

        if method.lower() == "ols":
            beta = OLS(X_train_scaled,z_train_scaled)
            zpredict = X_test_scaled@beta

        elif method.lower() == "ridge":
            beta = Ridge(X_train_scaled,z_train_scaled,lam=lam)
            zpredict = X_test_scaled@beta

        elif method.lower() == "lasso":
            model = Lasso(X_train_scaled,z_train_scaled,lam=lam)
            zpredict = model.predict(X_test_scaled)
            beta = model.coef_

        beta_parameters[degree-1] = beta
        polydegree[degree-1] = degree
        MSE_test[degree-1] = MSE(z_test_scaled,zpredict)
        R2_test[degree-1] = R2(z_test_scaled,zpredict)

    beta_std = np.sqrt(np.diag(np.var(z_train_scaled)*np.linalg.pinv(X_train_scaled.T @ X_train_scaled)))


    if plot:
        plt.style.use('seaborn-v0_8')

        for i in range(maxdegree):
            plt.scatter(np.arange(len(beta_parameters[i])),beta_parameters[i],label=f"n = {i+1}")
            plt.plot(np.arange(len(beta_parameters[i])),beta_parameters[i],'--')

        plt.xlabel(r"$j$")
        plt.ylabel(r"$\beta_j$")
        plt.xticks(np.arange(0, 21, 1.0))
        plt.legend()
        plt.show()

        for i in range(len(beta_std)):
            plt.errorbar(i,beta_parameters[-1][i],yerr=beta_std[i],marker='o',color="midnightblue")

        plt.xlabel(r"$j$")
        plt.ylabel(r"$\beta_j$")
        plt.xticks(np.arange(0, 21, 1.0))
        plt.show()

        plt.subplot(211)
        plt.plot(polydegree,MSE_test,color="midnightblue")
        plt.xticks(polydegree)
        plt.ylabel("MSE")


        plt.subplot(212)
        plt.plot(polydegree,R2_test,color="midnightblue")
        plt.xlabel("Complexity")
        plt.ylabel(f"$R^2-Score$")
        plt.xticks(polydegree)

        plt.show()


    return polydegree,MSE_test,R2_test
