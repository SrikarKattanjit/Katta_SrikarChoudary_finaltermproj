#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[8]:


def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate performance metrics based on true labels and predicted labels.

    Parameters:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels

    Returns:
    - A dictionary containing accuracy, precision, recall, F1-score, FPR, and FNR
    """
    # Initialize counters for true positives, false positives, true negatives, and false negatives
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Calculate TP, FP, TN, FN
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            true_positives += 1
        elif true == 0 and pred == 1:
            false_positives += 1
        elif true == 0 and pred == 0:
            true_negatives += 1
        elif true == 1 and pred == 0:
            false_negatives += 1

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) != 0 else 0
    fnr = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "FPR": fpr,
        "FNR": fnr
    }

# Example usage:
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 1, 1, 0, 0, 1, 0, 0]

metrics = calculate_performance_metrics(y_true, y_pred)
print(f'Accuracy: {metrics["accuracy"]:.2f}')
print(f'Precision: {metrics["precision"]:.2f}')
print(f'Recall: {metrics["recall"]:.2f}')
print(f'F1 Score: {metrics["f1_score"]:.2f}')
print(f'False Positive Rate (FPR): {metrics["FPR"]:.2f}')
print(f'False Negative Rate (FNR): {metrics["FNR"]:.2f}')


# 

# In[9]:


from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Initialize 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the classifier
clf = DecisionTreeClassifier(random_state=42)

# To store metrics for each fold
metrics = []

def calculate_tss(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return sensitivity - (1 - specificity)

def calculate_hss(tp, tn, fp, fn):
    denominator = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return 2 * ((tp * tn - fn * fp) / denominator) if denominator != 0 else 0

# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate TSS and HSS
    tss = calculate_tss(tp, tn, fp, fn)
    hss = calculate_hss(tp, tn, fp, fn)
    
    # Store metrics
    metrics.append((tp, tn, fp, fn, tss, hss))

# Convert metrics list to a NumPy array for easy averaging
metrics_np = np.array(metrics)

# Calculate average metrics across all folds
avg_metrics = np.mean(metrics_np, axis=0)

print("Average metrics across all 10 folds:")
print(f"TP: {avg_metrics[0]:.0f}, TN: {avg_metrics[1]:.0f}, FP: {avg_metrics[2]:.0f}, FN: {avg_metrics[3]:.0f}, TSS: {avg_metrics[4]:.2f}, HSS: {avg_metrics[5]:.2f}")


# LSTM

# In[ ]:





# In[2]:


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Because LSTMs expect 3D input (samples, timesteps, features), 
# and our data is not naturally sequential, we'll treat each feature as a timestep.
# This is just for demonstration and not a recommended approach for this dataset.

# Normalize the data to be between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X to be 3D
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)

# Convert labels to categorical (one-hot encoding)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Build the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=100, batch_size=64, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Test Accuracy: {accuracy:.2f}')


# Support Vector Machines

# In[3]:


# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an SVM classifier. Here we use a pipeline to first standardize the dataset.
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Decision Trees

# In[4]:


# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# KNN, K-Nearest Neighbor

# In[5]:


# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a KNN classifier. Here we use a pipeline to first standardize the dataset.
# Standardization is important for KNN because it is distance-based.
knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

# Train the classifier
knn_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Algorithm (Conv1D)

# In[7]:


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X to have the shape (samples, timesteps, features) for Conv1D
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)

# Convert labels to one-hot encoding
y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)

# Define the Conv1D model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_ohe, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_ohe)
print(f'Test Accuracy: {accuracy:.2f}')

