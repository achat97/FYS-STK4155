import numpy as np



def GD_analytical(X,y,betainit,niterations,eta,lmbda=0,gamma=0):
    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta 
                    v = gamma*v-eta*g
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta
                    



def GD_analytical_Adagrad(X,y,betainit,niterations,eta,delta=1e-7,lmbda=0,gamma=0):
    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    r = np.zeros(beta.shape)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta 
                    
                    r += g*g
                    v = gamma*v-(eta/(delta+np.sqrt(r)))*g
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta





def GD_analytical_RMSprop(X,y,betainit,niterations,eta,rho=0.9,delta=1e-6,lmbda=0,gamma=0):
    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    r = np.zeros(beta.shape)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta 
                    
                    r = rho*r+(1-rho)*g*g
                    v = gamma*v-(eta/(np.sqrt(delta+r)))*g
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta


def GD_analytical_ADAM(X,y,betainit,niterations,eta=0.001,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0):
    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)

    r = np.zeros(beta.shape)
    s = np.zeros(beta.shape)


    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta 
                    
                    s = rho1*s+(1-rho1)*g
                    r = rho2*r+(1-rho2)*g*g

                    shat = s/(1-rho1**(i+1))
                    rhat = r/(1-rho2**(i+1))

                    v = -eta*(shat/(delta+np.sqrt(rhat)))
                    beta += v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta