import numpy as np
from functions import *



def SGD_analytical(X,y,betainit,nepochs,nbatches,eta,lmbda=0,gamma=0):
    np.random.seed(2014)

    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))

    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)

    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
                
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = 2.0/M*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                v = gamma*v-eta*g
                beta += v
    
    return beta




def SGD_analytical_scheduler(X,y,betainit,nepochs,nbatches,t0,t1,lmbda=0,gamma=0):
     
    np.random.seed(2014)

    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))

    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    eta_j = t0/t1
    j = 0
    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
                
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = 2.0/M*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                v = gamma*v-eta_j*g
                beta += v

                t = epoch*nbatches+k
                eta_j = step_length(t,t0,t1)
                j += 1
    
    return beta




def SGD_analytical_Adagrad(X,y,betainit,nepochs,nbatches,eta,delta=1e-7,lmbda=0,gamma=0):
     
    np.random.seed(2014)

    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))
    r = np.zeros(beta.shape)


    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    

    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
                
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = 2.0/M*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                r += g*g
                v = gamma*v-(eta/(delta+np.sqrt(r)))*g
                beta += v
    


    return beta

    

def SGD_analytical_RMSprop(X,y,betainit,nepochs,nbatches,eta,rho,delta=1e-7,lmbda=0,gamma=0):
     
    np.random.seed(2014)

    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))
    r = np.zeros(beta.shape)


    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    

    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
                
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = 2.0/M*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                r = rho*r+(1-rho)*g*g
                v = gamma*v-(eta/(np.sqrt(delta+r)))*g
                beta += v
    


    return beta



def SGD_analytical_ADAM(X,y,betainit,nepochs,nbatches,eta=0.001,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0):
     
    np.random.seed(2014)

    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))

    r = np.zeros(beta.shape)
    s = np.zeros(beta.shape)


    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    
    t = 0
    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
            
            t+=1
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = 2.0/M*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta

                s = rho1*s+(1-rho1)*g
                r = rho2*r+(1-rho2)*g*g

                shat = s/(1-rho1**(t))
                rhat = r/(1-rho2**(t))
                
                v = -eta*(shat/(delta+np.sqrt(rhat)))
                beta += v
    


    return beta







