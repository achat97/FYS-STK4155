from functions import *


def biasVariance(N,z_tuning,maxdegree,B,method,plot=True,lam=0):

    x,y,z = Data(N,z_tuning)
    z = z.ravel()

    polydegree = np.zeros(maxdegree)
    MSE_test = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    var = np.zeros(maxdegree)

    X = create_X(x, y, maxdegree)
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)

    for degree in range(1,maxdegree+1):

        X_train_split,X_test_split = features(X_train,degree),features(X_test,degree)
        X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled = centering(X_train_split,X_test_split,z_train,z_test)
        z_train_scaled,z_test_scaled = z_train_scaled[:,None],z_test_scaled[:,None]

        z_pred = bootstrap(X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled,B,method,lam)

        polydegree[degree-1] = degree
        MSE_test[degree-1] = np.mean( np.mean((z_test_scaled - z_pred)**2, axis=1, keepdims=True) )
        bias[degree-1] =np.mean( (z_test_scaled - np.mean(z_pred, axis=1, keepdims=True))**2 )
        var[degree-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )


        print('Polynomial degree:', degree)
        print('Error:', MSE_test[degree-1])
        print('Bias^2:', bias[degree-1])
        print('Var:', var[degree-1])
        print('{} >= {} + {} = {}'.format(MSE_test[degree-1], bias[degree-1], var[degree-1], bias[degree-1]+var[degree-1]))
        print("_______________________________________________________________")


    if plot:
        plt.style.use('seaborn-v0_8')
        plt.plot(polydegree,MSE_test,"-o",color="midnightblue",label=r"$MSE$")
        plt.plot(polydegree,bias,"-^",color="darkgoldenrod",label=r"$Bias$")
        plt.plot(polydegree,var,"-s",color="seagreen",label=r"$Variance$")
        plt.xlabel("Complexity")
        plt.xticks(polydegree)
        plt.legend()
        plt.show()

    return polydegree,MSE_test
