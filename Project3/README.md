# *Project 3*

* Report is found in `Project`
* All scripts are found in `Src`
* All figures used in the report can be found in `Figures`


## **Packages**
The required packages are 
* `numpy`
* `matplotlib`
* `sklearn`
* `Tensorflow`
* `Seaborn`


## **Execution**
Run `heat_equation.py` to generate the results related to the analytical solution and the finite difference solution. `NN_main.py`  will generate the results related to the solution using the neural network.  `NN_PDE.py` contains the setup of the neural network and is used as a library. 

`elu_model.keras` is the trained model we are mainly using throughout the project. It uses 6 hidden layers, with 500 neurons in each layer and ELU as the activation, trained for 5000 epochs. Other models are also included `elu_model_10k.keras` is the model trained for 10,000 epochs but with the same architecture.  `elu_model_100neurons.keras` is a model with the same setup as the first, but with 100 neurons instead of 500.