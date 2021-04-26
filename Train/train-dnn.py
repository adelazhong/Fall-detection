import sys
from tensorflow import keras as K
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import math
import pandas as pd
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
    # Add a column of target to encode the target variable
    F_Detection['target'] = F_Detection[target_var].apply(lambda x: Class_dict[x])
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(F_Detection['target'])
    y_bin_labels = []  
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        F_Detection['y' + str(i)] = transformed_labels[:, i]
    # Separate the data into train and validation 7:3
    train_x, val_x, train_y, val_y = train_test_split(F_Detection[features], F_Detection[y_bin_labels], \
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_x, val_x, train_y, val_y, Class_dict

    # Data normalization
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

if __name__ == '__main__':
    
    # Load the CSV Data
    CSV_path = "../Data/Feature_train_data.csv"
    train_x, val_x, train_y, val_y, Class_dict = load_data(CSV_path)

    # Modify Dataframe into array.
    train_x = train_x.values
    train_y = train_y.values
    val_x = val_x.values
    val_y = val_y.values
    pi = math.pi

    # Preprocessing of speed, angle and wideth_length ratio parameters.(Both train and validation)
    speed_val = normalization(val_x[:,0])
    angle_val = np.array(val_x[:,1])-pi/4
    l_w_val = np.array(val_x[:,2])-1
    val_x = np.vstack((speed_val,angle_val,l_w_val)).T
    speed_train = normalization(train_x[:,0])
    angle_train = np.array(train_x[:,1])-pi/4
    l_w_train = np.array(train_x[:,2])-1
    train_x = np.vstack((speed_train,angle_train,l_w_train)).T
    
    # DNN model
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam(learning_rate=0.0005)
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=5, input_dim=3, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=["accuracy"])

    # Train the model
    b_size = 4
    max_epochs = 500
    print("Starting training ")
    _history = model.fit(train_x,train_y,batch_size=b_size,epochs=max_epochs,validation_data=(val_x,val_y), verbose=1)
    
    # Save the model
    model.save("../Models/dnn_model_PR.h5")
    print("Training finished \n")

    # Visualize the accuracy result and save it.
    plt.style.use("ggplot")
    plt.figure()
    N = max_epochs
    plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")
    plt.plot(np.arange(0,N),_history.history["val_loss"],label="val_loss")
    plt.plot(np.arange(0,N),_history.history["accuracy"],label="train_acc")
    plt.plot(np.arange(0,N),_history.history["val_accuracy"],label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("result-dnn.png")
    plt.show()

    # Visualize the Precision and Recall result and save it.
    lr_probs = []
    lr_probs = model.predict_proba(val_x)
    lr_probs = lr_probs[:, 1]
    val_y = np.squeeze(val_y)
    lr_precision, lr_recall, _ = precision_recall_curve(val_y, lr_probs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(lr_recall, lr_precision,label ="Logistic")
    plt.title("Percision and Recall")
    plt.xlabel("Recall")
    plt.ylabel("Percision")
    plt.legend(loc="best")
    plt.savefig("dnn-P-R.png")
    plt.show()
    

