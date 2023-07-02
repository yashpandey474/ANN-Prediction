# Churn Prediction Neural Network
This project implements a neural network model for churn prediction using customer data. The goal is to predict whether a customer is likely to churn (i.e., leave) a service or not based on various features. Built as part of the Deep Learning A-Z course on Udemy.

# Dataset
The dataset used for this project is "Churn_Modelling.csv". It contains customer information such as age, gender, credit score, balance, etc., along with a label indicating whether the customer churned or not.

# Preprocessing
Before training the neural network, the dataset undergoes preprocessing steps:

1. Categorical features (country and gender) are encoded using label encoding and one-hot encoding techniques, respectively.
2. The dataset is split into training and test sets using a 80:20 ratio.
3. Feature scaling is applied to standardize the numerical features using the StandardScaler from scikit-learn.

# Neural Network Architecture
The neural network architecture consists of an input layer, two hidden layers, and an output layer.

1. The input layer has the same number of neurons as the number of features in the dataset.
2. The two hidden layers contain 6 neurons each and use the ReLU activation function.
3. The output layer has a single neuron with a sigmoid activation function, producing a probability of churn.

# Training
The neural network is trained on the training set using the Adam optimizer and binary cross-entropy loss function. The model is trained for 200 epochs with a batch size of 32.

# Results
After training, the model achieves an accuracy of approximately 86.5% on the training set.

# Usage
To run the code, make sure you have the necessary dependencies installed. You can then execute the code using any Python environment.

# Install the required dependencies
pip install numpy pandas tensorflow scikit-learn

# Run the code
1. python churn_prediction_nn.py

Acknowledgements and Credits
I implemented this project as taught in the Udemy course "Deep Learning A-Z™: Hands-On Artificial Neural Networks" by Kirill Eremenko and Hadelin de Ponteves.
