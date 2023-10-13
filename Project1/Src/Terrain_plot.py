from functions import *


# Load the terrain
terrain,x,y,z = Data_terrain("SRTM_data_Morocco.tif",30,600,2600)
z_scaled = (z-np.mean(z.ravel()))


plt.figure()
plt.imshow(terrain, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.figure()
plt.imshow(z, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.style.use('seaborn-v0_8')
surf = ax.plot_surface(x, y, z, cmap=cm.gist_earth,linewidth=0, antialiased=False)

# Customize axis.
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()



def crossValidation(X,z,k,method,lam=0):



    """
    Calculate mean square error of model using k-fold cross validation without shuffling

    Args:
        X (ndarray): design matrix
        z (ndarray): target data
        k (int): number of folds
        method (string): regression method (OLS, lasso or ridge)

    Returns:
        error (float): mean square error
        error2 (float): R2-score

    """


    nind = np.shape(X)[0]
    ind = np.arange(0,nind,1)
    train_ind = []
    test_ind = []
    R = nind%k
    n = int(np.floor(nind/k))


    start = 0
    stop = n
    for i in range(k):

        if R>0:
            test_ind.append(ind[start:stop+1])
            train_ind.append(np.concatenate((ind[:start],ind[stop+1:])))
            R-=1
            start+=(n+1)
            stop+=(n+1)


        else:
            test_ind.append(ind[start:stop])
            train_ind.append(np.concatenate((ind[:start],ind[stop:])))
            start+=n
            stop+=n


    if method.lower() == "ols":

        error = np.zeros(k)
        errorR2 = np.zeros(k)
        betas = []

        i=0
        for train_index, test_index in zip(train_ind,test_ind):
            Xtrain = X[train_index,:]
            ztrain = z[train_index]

            Xtest = X[test_index,:]
            ztest = z[test_index]

            Xtrain_scaled,Xtest_scaled,ztrain_scaled,ztest_scaled = centering(Xtrain,Xtest,ztrain,ztest)


            beta = OLS(Xtrain_scaled, ztrain_scaled)
            betas.append(beta)
            zpred = (Xtest_scaled@beta).ravel()

            error[i] = MSE(ztest_scaled.ravel(),zpred)
            errorR2[i] = R2(ztest_scaled.ravel(),zpred)
            i += 1




    elif method.lower() == "ridge":

        kfold = KFold(n_splits = k)
        error = np.zeros(k)
        errorR2 = np.zeros(k)

        i=0
        for train_index, test_index in kfold.split(X):
            Xtrain = X[train_index,:]
            ztrain = z[train_index]

            Xtest = X[test_index,:]
            ztest = z[test_index]

            Xtrain_scaled,Xtest_scaled,ztrain_scaled,ztest_scaled = centering(Xtrain,Xtest,ztrain,ztest)

            beta = Ridge(Xtrain_scaled, ztrain_scaled,lam=lam)
            zpred = (Xtest_scaled@beta).ravel()

            error[i] = MSE(ztest_scaled.ravel(),zpred)
            errorR2[i] = R2(ztest_scaled.ravel(),zpred)
            i += 1


    elif method.lower() == "lasso":

        kfold = KFold(n_splits = k)
        error = np.zeros(k)
        errorR2 = np.zeros(k)

        i=0
        for train_index, test_index in kfold.split(X):
            Xtrain = X[train_index,:]
            ztrain = z[train_index]

            Xtest = X[test_index,:]
            ztest = z[test_index]

            Xtrain_scaled,Xtest_scaled,ztrain_scaled,ztest_scaled = centering(Xtrain,Xtest,ztrain,ztest)


            model = Lasso(X,z,lam=lam)
            zpred = model.predict(Xtest_scaled)

            error[i] = MSE(ztest_scaled.ravel(),zpred)
            errorR2[i] = R2(ztest_scaled.ravel(),zpred)
            i += 1


    return np.mean(error[1:-1]), np.mean(errorR2[1:-1]), np.mean(betas, axis=0)


#plot of optimal model
X = create_X(x, y, 10)
m,r,betas = crossValidation(X,z.ravel(),10,'ols',lam=0)


zz = X * betas  
zz = zz.sum(axis=1)
zz = zz.reshape(30, 30)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.style.use('seaborn-v0_8')
surf = ax.plot_surface(x, y, zz, cmap=cm.gist_earth,linewidth=0, antialiased=False)

# Customize axis.
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

