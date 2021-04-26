import sys
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import matplotlib.pylab as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load CSV Feature Data and seperated it into train and validation
def load_data(CSV_FILE_PATH):
    F_Detection = pd.read_csv(CSV_FILE_PATH)
    target_var = 'class'  # Target variable
    # Feature of the datasets
    features = list(F_Detection.columns)
    features.remove(target_var)
    # Target value class
    Class = F_Detection[target_var].unique()
    Class_dict = dict(zip(Class, range(len(Class))))
    # Add a column of Target to encode the target variable
    F_Detection['target'] = F_Detection[target_var].apply(lambda x: Class_dict[x])
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(F_Detection['target'])
    y_bin_labels = []  
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        F_Detection['y' + str(i)] = transformed_labels[:, i]

    # Get the top 2000 frames from CSV file for testing                               
    test_x = F_Detection[features].head(2000)
    test_y = F_Detection[y_bin_labels].head(2000)
    return  test_x, test_y, Class_dict

# Data Normalization 
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range




if __name__ == '__main__':
    
    # Load the Pose test file for RNN model
    test_x,  test_y, Class_dict = load_data("../Data/Feature_test_data.csv")

    # Load the model
    model = K.models.load_model("../Models/dnn_model_PR.h5")
    
    # Modify the Dataframe into array
    test_x = test_x.values
    test_y = test_y.values

    # Preprocessing of speed, angle and wideth_length ratio parameters.
    pi = math.pi
    speed_test = normalization(test_x[:,0])
    angle_test = np.array(test_x[:,1])-pi/4
    l_w_test = np.array(test_x[:,2])-1
    test_x = np.vstack((speed_test,angle_test,l_w_test)).T

    
    # Evaluate the model 
    eval = model.evaluate(test_x, test_y, verbose=0)
    print("Evaluation on test data: accuracy = %0.2f%% \n" \
          % (eval[1] * 100) )

    # Export the CSV result files for Video Visualization.
    np.set_printoptions(precision=4)
    unknown = np.array(test_x, dtype=np.float32)
    predicted = model.predict(unknown)
    FD_dict = {v:k for k,v in Class_dict.items()}
    ans = []
    for i in range(len(predicted)):
        ans.append(FD_dict[np.argmax(predicted[i])])

    # Save the evaluation result as CSV file.
    np.savetxt('dnn_result.csv', ans, delimiter = ',')