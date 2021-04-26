import sys
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Load CSV Pose Data and seperated it into train and validation
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
    
    # Get the top 2500 frames from CSV file for train and last 2500 frames for validation
    train_x = F_Detection[features].head(2500)
    train_y = F_Detection[y_bin_labels].head(2500)
    val_x = F_Detection[features].tail(2500)
    val_y = F_Detection[y_bin_labels].tail(2500)

    return train_x, val_x, train_y, val_y, Class_dict

if __name__ == '__main__':
    
    #Load the CSV file
    CSV_File = "../Data/Pose_train_data.csv"
    train_x, val_x, train_y, val_y, Class_dict = load_data(CSV_File)

    # Verify Dataframe into numpy array
    train_x = train_x.values
    train_y = train_y.values
    val_x = val_x.values
    val_y = val_y.values
    Class_dict = Class_dict.values
    train_x = train_x.reshape(len(train_x), 1, 75)
    train_y = train_y.reshape(len(train_y), 1, len(train_y[0]))
    val_x = val_x.reshape(len(val_x), 1, 75)
    val_y = val_y.reshape(len(val_y), 1, len(val_y[0]))
    
    # Define the training models. Using Bidirectional LSTM
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam(learning_rate=0.00001)
    model = K.models.Sequential()
    model.add(
        K.layers.Bidirectional(K.layers.LSTM(64, return_sequences=True), input_shape=(1, 75))
    )
    model.add(K.layers.Bidirectional(K.layers.LSTM(32)))
    model.add(K.layers.Dense(2, activation = 'softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics = ['accuracy'])

    # Train the model
    b_size = 16
    max_epochs = 1000
    print("Starting training ")
    _history = model.fit(train_x,train_y,batch_size=b_size,epochs=max_epochs,validation_data=(val_x,val_y), verbose=1)

    # Save the model
    model.save("../Models/rnn_model_PR.h5")
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
    plt.savefig("result.png")
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
    plt.savefig("P_R_RNN.png")
    plt.show()
    
