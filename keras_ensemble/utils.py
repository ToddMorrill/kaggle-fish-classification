"""
Helper functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

def count_files(path):
    """Counts the number of files in a directory with nested folders
    Args:
        path (str) : path of the parent directory containing nested folders
    """
    counter = 0
    for directory in os.listdir(path):
        if os.path.isdir(path+directory):
            counter+= len(os.listdir(path+directory))

    return counter

def log_string(path, string):
    """Logs a string in a text file
    Args:
        path (str) : path of the log file
        string (str) : string to be logged
    """
    with open(path, "a") as f:
        f.write(string+"\n")
        
def get_ground_truth_labels(test_path, num_test_samples, class_labels):
    """Provides ground truth labels for testing
    
    Args:
        test_path (str) : path to the test directory with nested class folders
        num_test_samples (int) : number of test samples
        class_labels (list) : ordered list of class_labels
    
    Returns:
        numpy.array : array of ground truth class labels
    """
    ground_truth_labels = [0]*num_test_samples
    start_index = 0  
    for index, label in enumerate(class_labels):
        end_index = start_index + len(os.listdir(test_path+label))
        ground_truth_labels[start_index:end_index] = [label]*(end_index-start_index)
        start_index = end_index
    return np.array(ground_truth_labels)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        cm = np.round_(cm, decimals=2)
    else:
        pass
        #print('Confusion matrix')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')