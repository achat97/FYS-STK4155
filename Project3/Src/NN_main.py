from NN_PDE import *
from heat_equation import *
import os
from sklearn.model_selection import train_test_split
import seaborn as sns


#Remove all the these paths to save figures where its run or create own path
save_directory = '/Users/achrafatifi/Documents/FYS-STK4155/FYS-STK4155/Project3/Src/' 
parent_directory = os.path.join(save_directory, '..')
figures_directory = os.path.join(parent_directory, 'Figures')

x = np.linspace(0,1,25)
t = np.linspace(0,1,25)
x, t = np.meshgrid(x, t)
x, t = x.ravel(),t.ravel()
xtrain, xtest, ttrain, ttest = train_test_split(x,t,test_size=0.2,random_state=40)

xtest = tf.convert_to_tensor(xtest.reshape(-1, 1), dtype=tf.float32)
ttest = tf.convert_to_tensor(ttest.reshape(-1, 1),dtype=tf.float32)

def heatmap_ln():
    """
    Creates a heatmap using a neural network for different numbers of hidden layers and neurons of mean absolute error
    """

    nlayers = [2,4,6,8,10]
    nneurons = [10,50,100,500,1000]
    maegrid = np.zeros((len(nlayers),len(nneurons)))


    for i in range(len(nlayers)):
        for j in range(len(nneurons)):
            print(f'layers: {nlayers[i]} – neurons: {nneurons[j]}')

            layerinp = []
            for k in range(nlayers[i]):
                layerinp.append(nneurons[j])
            
            model,opt = create_neural_network(layerinp,'sigmoid','glorot')
            loss_iterations = fit(model,opt,xtrain,ttrain,len(xtrain),5000)

            yanalytical =  heat_analytical(xtest,ttest)
            ypred = g_trial(model,xtest,ttest,train=False)

            mae = tf.keras.losses.MeanAbsoluteError()   
            print('MAE: ',mae(yanalytical, ypred).numpy())
            maegrid[i][j] = mae(yanalytical, ypred).numpy() 
            print()



    xticks = [f"${neuron}$" for neuron in nneurons]
    yticks = [f"${layer}$" for layer in nlayers]

    fig, ax = plt.subplots(layout='constrained', figsize=(13,11))

    heatmap_ln = sns.heatmap(maegrid, annot=True, cmap="YlOrRd",yticklabels=yticks,xticklabels=xticks, annot_kws={"fontsize":22} ,ax=ax)

    cbar_ln = heatmap_ln.collections[0].colorbar
    cbar_ln.ax.tick_params(labelsize=19)  
    cbar_ln.set_label('MAE', fontsize=19)

    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel("L-1",fontsize=19)
    ax.set_xlabel("$N_l$",fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=19)

    plt.savefig(os.path.join(figures_directory, 'Heatmap_ln.png'))


def act_iter():
    """
    Plots a neural network with 6 hidden layers with 500 neurons each for differenct activation function (sigmoid,tanh,relu,elu)
    """

    it = 2000
    model_log,opt_log = create_neural_network([500,500,500,500,500,500],'sigmoid','glorot')
    model_tanh,opt_tanh = create_neural_network([500,500,500,500,500,500],'tanh','glorot')
    model_relu,opt_relu = create_neural_network([500,500,500,500,500,500],'relu','he')
    model_elu,opt_elu = create_neural_network([500,500,500,500,500,500],'elu','he')

    print('Logistic:')
    loss_iterations_log = fit(model_log,opt_log,xtrain,ttrain,len(xtrain),it)
    print()
    print('Tanh:')
    loss_iterations_tanh = fit(model_tanh,opt_tanh,xtrain,ttrain,len(xtrain),it)
    print()
    print('ReLU:')
    loss_iterations_relu = fit(model_relu,opt_relu,xtrain,ttrain,len(xtrain),it)
    print()
    print('ELU:')
    loss_iterations_elu = fit(model_elu,opt_elu,xtrain,ttrain,len(xtrain),it)

    x = np.arange(1,it+1)

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))

    ax.plot(x,loss_iterations_log,label='Logistic',linewidth=9,color='midnightblue')
    ax.plot(x,loss_iterations_tanh,label='Tanh',linewidth=9,color='red')
    ax.plot(x,loss_iterations_relu,label='ReLU',linewidth=9,color='green')
    ax.plot(x,loss_iterations_elu,label='ELU',linewidth=9,color='orange')

    ax.set_xlabel('Iterations',fontsize=28)
    ax.set_ylabel('Loss',fontsize=28)
    ax.set_ylim(0,30)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)

    plt.savefig(os.path.join(figures_directory,'act_loss.png'))


def sig_tanh():
    """
    Plots the logistic and hyperbolic tangent functions
    """

    x = np.linspace(-5,5,1000)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(x,sigmoid,label=f'$\sigma(x)$',linewidth=9,color='midnightblue')
    ax.plot(x,tanh,label=f'tanh$(x)$',linewidth=9,color='red')

    ax.set_xlabel('x',fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)

    plt.savefig(os.path.join(figures_directory,'sig_tanh.png'))


def Heatmap_err():
    """
    Heatmap of the pointwise error of the trained neural network
    """

    model_elu = tf.keras.models.load_model('elu_model.keras')
    n = 100
    x_ = np.linspace(0,1,n)
    t_  = np.linspace(0,1,n)
    x_, t_ = np.meshgrid(x_, t_)
    x_in, t_in = x_.ravel().reshape(-1, 1),t_.ravel().reshape(-1, 1)
    
    X = tf.convert_to_tensor(x_in, dtype=tf.float32)
    T = tf.convert_to_tensor(t_in,dtype=tf.float32)
    yanalytical =  heat_analytical(X,T)
    ypred_elu = g_trial(model_elu,X,T,train=False)
    error = tf.reshape(np.abs(yanalytical-ypred_elu), [n, n]).numpy()

    mae = tf.keras.losses.MeanAbsoluteError()   
    mae_result = mae(yanalytical, ypred_elu).numpy()

    nlevels = np.linspace(np.min(error), np.max(error),60)

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))
    heatmap = plt.contourf(x_,t_,error,levels=nlevels,cmap="nipy_spectral")
    cb = plt.colorbar(heatmap,format="{x:.4f}")
    cb.ax.tick_params(labelsize=28)
    cb.ax.set_ylabel(r'|$y-\tilde{y}$|', fontsize=28,rotation=90,labelpad=15)
    cb.ax.locator_params(nbins=10)
    ax.set_xlabel("$x$",fontsize=28)
    ax.set_ylabel("$t$",fontsize=28)
    ax.set_title(f'MAE={mae_result:.5f}',fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    plt.savefig(os.path.join(figures_directory,'heatmap_error.png'))


def plot_heat_NN(dx,t,figname):
    """
    Plots the analytical solution of the heat equation and the solution using the FTCS method and the neural network for a given time

    Args:
        t (float): time
        dx (float): step size in space
        figname (str): figure name
    """

    x = np.arange(0,1+dx,dx).reshape(-1,1)

    model_elu = tf.keras.models.load_model('elu_model.keras')
    X = tf.convert_to_tensor(x.reshape(-1,1), dtype=tf.float32)
    T = tf.convert_to_tensor(np.full(x.reshape(-1,1).shape,t),dtype=tf.float32)

    u_num= u_finite(t,dx)
    u_analytical = np.sin(x*np.pi)*np.exp(-t*(np.pi)**2 )
    u_NN = g_trial(model_elu,X,T,train=False)
    


    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(x,u_num,label="FTCS",linestyle='-.',linewidth=9,color='midnightblue')
    ax.plot(x,u_analytical,label='Analytical',linewidth=9,color='green')
    ax.plot(x,u_NN,label='NN',linewidth=9,linestyle='--',color='red')
    ax.set_xlabel('x',fontsize=28)
    ax.set_ylabel('Temperature',fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)
    ax.set_title(f'$\Delta x$={dx}, $t$={t}',fontsize=28)

    plt.savefig(os.path.join(figures_directory,figname+'.png'))


def error_comparison(dx,figname):
    """
    Plots the mean absolute error of the solution using the FTCS method and a neural network against time

    Args:
        t (float): time
        dx (float): step size in space
        figname (str): figure name
    """

    x = np.arange(0, 1+dx, dx).reshape(-1, 1)
    dt = (dx**2)/2
    time = np.arange(0, 1  + dt, dt).reshape(-1, 1)

    u_num = np.zeros((len(x),len(time)))
    u_NN = np.zeros((len(x),len(time)))
    u_analytical_NN = np.zeros((len(x),len(time)))

    model_elu = tf.keras.models.load_model('elu_model.keras')
    X = tf.convert_to_tensor(x, dtype=tf.float32)

    for i, t in enumerate(time):
        print('t: ',t)
        T = tf.convert_to_tensor(np.full(x.shape,t),dtype=tf.float32)
        u_num[:, i] = u_finite(t, dx)[:,0]
        u_NN[:,i] =  g_trial(model_elu,X,T,train=False).numpy().ravel()
        u_analytical_NN[:,i] =  heat_analytical(X,T).numpy().ravel()

    t, x = np.meshgrid(time, x)
    u_analytical = np.sin(x*np.pi)*np.exp(-t*(np.pi)**2 )

    mae_num = np.zeros(len(time))
    mae_NN = np.zeros(len(time))
    mae = tf.keras.losses.MeanAbsoluteError()   
    for i in range(len(time)):
        mae_num[i] = np.mean(np.abs((u_analytical[:,i]-u_num[:,i])))
        mae_NN[i] = mae(u_analytical_NN[:,i], u_NN[:,i]).numpy()

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(layout='constrained', figsize=(14,11))
    ax.plot(time,mae_num,linewidth=9,color='midnightblue',label='FTCS')
    ax.plot(time,mae_NN,linewidth=9,color='red',label='NN')
    ax.set_xlabel('t',fontsize=28)
    ax.set_ylabel('MAE',fontsize=28)
    ax.tick_params(axis='both', which='both', labelsize=28)
    ax.set_title(f'$\Delta x$={dx}',fontsize=28)
    ax.yaxis.get_children()[1].set_size(28)
    ax.legend(fontsize=28)

    plt.savefig(os.path.join(figures_directory,figname+'.png'))


def plot_heat3D_NN():
    """
    Plots the solution approximated by the neural network for the heat equation for x in [0,1] and t in [0,1] in 3D
    """

    model_elu = tf.keras.models.load_model('elu_model.keras')
    n = 1001
    x_ = np.linspace(0,1,n)
    t_  = np.linspace(0,1,n)
    
    x_, t_ = np.meshgrid(x_, t_)
    x_in, t_in = x_.ravel().reshape(-1, 1),t_.ravel().reshape(-1, 1)
    
    X = tf.convert_to_tensor(x_in, dtype=tf.float32)
    T = tf.convert_to_tensor(t_in,dtype=tf.float32)
    ypred_elu = tf.reshape(g_trial(model_elu,X,T,train=False), [n, n]).numpy()

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(14,11))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(t_, x_, ypred_elu, cmap='cividis',linewidth=0, antialiased=False)

    # Customize the plot
    ax.set_xlabel('t',fontsize=32,labelpad=15)
    ax.set_ylabel('x',fontsize=32,labelpad=15)
    ax.set_zlabel('Temperature',fontsize=32,labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=28)
    fig.tight_layout()
    plt.savefig(os.path.join(figures_directory,'analytical3D_NN.png'))

"Uncomment to run"
#heatmap_ln()
#act_iter()
#sig_tanh()
#Heatmap_err()
#error_comparison(1/10,'FTCS_NN_largestep')
#error_comparison(1/100,'FTCS_NN_smallstep')
#plot_heat_NN(1/100,0.8,'all_t08')
#plot_heat_NN(1/100,0.04,'all_t004')
#plot_heat3D_NN()