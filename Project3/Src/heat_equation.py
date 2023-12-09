import numpy as np
import matplotlib.pyplot as plt
import os

#Remove all the these paths to save figures where its run or create own path
save_directory = '/Users/achrafatifi/Documents/FYS-STK4155/FYS-STK4155/Project3/Src/' 
parent_directory = os.path.join(save_directory, '..')
figures_directory = os.path.join(parent_directory, 'Figures')


def u_finite(t,dx):
    """
    Solves the 1D heat equation using the FTCS method for a given time

    Args:
        t (float): time
        dx (float): step size in space

    Returns:
        u (ndarray): values of the solved equation in space at the given time
    """
    
    x = np.arange(dx,1,dx).reshape(-1,1)
    N = x.shape[0]
    dt = (dx**2)/2
    v = np.sin(x*np.pi)
    

    
    A = np.zeros(shape=(N,N)) #Lager en nxn-matrise A med kun nuller
    A[0,0] = -2 #element a_(11)=2 i matrisen
    A[0,1] = 1 #element a_(12)=-1 i matrisen
    A[N-1,N-2] = 1 #element a_(N(N-1))=-1 i matrisen
    A[N-1,N-1] = -2 #element a_(NN)=2 i matrisen
    for i in range(1,N-1): #loop over radene til matrisen utenom første og siste rad
        A[i,i-1] = 1
        A[i,i] = -2
        A[i,i+1] = 1

    A = A/(dx**2)

    t_current = 0
    while t_current<t:
        v = np.matmul((np.identity(N)+dt*A),v)
        t_current+=dt
    
    u = np.zeros(N+2).reshape(-1,1)
    for i in range(1,N+1):
        u[i] = v[i-1]
    
    return u


def plot_heat(dx,t,figname):
    """
    Plots the analytical solution of the heat equation and the solution using the FTCS method for a given time

    Args:
        t (float): time
        dx (float): step size in space
        figname (str): figure name
    """

    x = np.arange(0,1+dx,dx).reshape(-1,1)

    u_num= u_finite(t,dx)
    u_analytical = np.sin(x*np.pi)*np.exp(-t*(np.pi)**2 )

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(x,u_num,label="FTCS",linestyle='-.',linewidth=9,color='midnightblue')
    ax.plot(x,u_analytical,label='Analytical',linewidth=9,color='red')
    ax.set_xlabel('x',fontsize=28)
    ax.set_ylabel('Temperature',fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)
    ax.set_title(f'$\Delta x$={dx}, $t$={t}',fontsize=28)

    plt.savefig(os.path.join(figures_directory,figname+'.png'))


def plot_heat3D():
    """
    Plots the analytical solution of the heat equation for x in [0,1] and t in [0,1] in 3D
    """

    x = np.linspace(0,1,1001)
    t = np.linspace(0,1,1001)

    x, t = np.meshgrid(x, t)

    u_analytical = np.sin(x*np.pi)*np.exp(-t*(np.pi)**2 )

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(14,11))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(t, x, u_analytical, cmap='cividis',linewidth=0, antialiased=False)

    # Customize the plot
    ax.set_xlabel('t',fontsize=32,labelpad=15)
    ax.set_ylabel('x',fontsize=32,labelpad=15)
    ax.set_zlabel('Temperature',fontsize=32,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=28)
    fig.tight_layout()
    plt.savefig(os.path.join(figures_directory,'analytical3D.png'))


def error_finite(dx,figname):
    """
    Plots the mean absolute error of the solution using the FTCS method against time


    Args:
        dx (float): step size in space
        figname (str): figure name
    """

    x = np.arange(0, 1+dx, dx).reshape(-1, 1)
    dt = (dx**2)/2
    time = np.arange(0, 1 + dt, dt).reshape(-1, 1)
    u_num = np.zeros((len(x),len(time)))

    for i, t in enumerate(time):
        print('t: ',t)
        u_num[:, i] = u_finite(t, dx)[:,0]

    t, x = np.meshgrid(time, x)
    u_analytical = np.sin(x*np.pi)*np.exp(-t*(np.pi)**2 )

    mae = np.zeros(len(time))
    for i in range(len(mae)):
        mae[i] = np.mean(np.abs((u_analytical[:,i]-u_num[:,i])))

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(time,mae,linewidth=9,color='midnightblue')
    ax.set_xlabel('t',fontsize=28)
    ax.set_ylabel('MAE',fontsize=28)
    ax.tick_params(axis='both', which='both', labelsize=28)
    ax.set_title(f'$\Delta x$={dx}',fontsize=28)
    ax.yaxis.get_children()[1].set_size(28)

    plt.savefig(os.path.join(figures_directory,figname+'.png'))


"Uncomment to run"
#plot_heat3D()
#error_finite(1/10,'largestep_error')
#error_finite(1/100,'smallstep_error')
#plot_heat(1/10,0.1,"largedx_smallt")
#plot_heat(1/10,0.9,"largedx_larget")
#plot_heat(1/100,0.1,"smalldx_smallt")
#plot_heat(1/100,0.9,"smalldx_larget")