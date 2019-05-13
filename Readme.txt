-------------------------------------------------------

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 30/10/18

------------------------------------------------------

*Copy the Training Set images path as command line argument


------------------------------------------------------
Dir://$ python Main.py PATH_TO_IMAGE_FOLDER

*Code asks for multiple inputs :-

1.Number of Hidden Layers in model
2.Activation function for model as sigmoid,tanh or relu
3.Enter dropouts for input layer and all hidden layers
4.Enter number of nodes in 1st Hidden Layer
5.Enter number of nodes in 2nd Hidden Layer and so on..
6.Batch Size for Gradient Descent
7.Learning Rate 
8.Number of iterations.


Output:-

1. Will give Error on train set and test set after each epoch
2. Plot the graph of validation and train error vs epoch.

-------------------------------------------------------

Sample Run :-

siddharth@Jarvis:~/5thSem/ML/NeuralNetworks$ python Main.py l3/steering/
Started Preprocessing of Data ........
PreProcessing Ended
Number of Hidden Layers in input : 2
Enter the name of Activation Function as sigmoid,tanh, or relu : sigmoid
Enter the Dropout fraction for 1 Layer : 0
Enter the Dropout fraction for 2 Layer : 0
Enter the Dropout fraction for 3 Layer : 0
Enter number of Nodes in 1 Hidden layer : 512
Enter number of Nodes in 2 Hidden layer : 64
Enter the Batch Size for Training : 64
Enter the Learning Rate : 0.05
Enter number of Epochs for training : 100
