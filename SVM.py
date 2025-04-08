import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define column names manually
column_names = ['c'+str(name) for name in range(10+30+1+1+98)]

# Load the dataset from a CSV file without header
#data = pd.read_csv('data_no_header.csv', header=None, names=column_names)
# Load the dataset from a CSV file
X_train = pd.read_csv('X_train_LDA.csv', header=None, names=column_names)
y_train = pd.read_csv('Y_train.csv', header=None, names=['target'])
#y_train = np.array(y_train)
y_train = y_train.values.ravel()
X_test =  pd.read_csv('X_test_LDA.csv', header=None, names=column_names)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Train the model on the training data
svm_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm_model.predict(X_test)

file = open('SVM_pred_LDA.csv', 'w', newline='')
file.write('chp2')
for i in range(12753):
    file.write('\n')
    file.write(str(y_pred[i]))
