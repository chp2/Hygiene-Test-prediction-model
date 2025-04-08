import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define column names manually
column_names = ['c'+str(name) for name in range(96305)]

# Load the dataset from a CSV file without header
#data = pd.read_csv('data_no_header.csv', header=None, names=column_names)
# Load the dataset from a CSV file
print('x_train loading')
X_train = pd.read_csv('X_train_TF_IDF.csv', header=None, names=column_names)
print('y_train loading')
y_train = pd.read_csv('Y_train.csv', header=None, names=['target'])
#y_train = np.array(y_train)
y_train = y_train.values.ravel()

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Train the model on the training data
svm_model.fit(X_train, y_train)

print('x_test loading')
for i in range(26):
    filename = 'X_test_TF_IDF' + str(i) + '.csv'
    X_test =  pd.read_csv(filename, header=None, names=column_names)
    print(i)
    # Make predictions on the testing data
    y_pred = svm_model.predict(X_test)

    y_pred_list = [[pred] for pred in y_pred]
    y_filename = 'SVM_Y_pred_TF_IDF' + str(i) + '.csv'
    file = open(y_filename, 'w', newline='')
    file.write(str(y_pred[0]))
    for i in range(1, len(y_pred_list)):
        #print(i)
        file.write('\n')
        file.write(str(y_pred[i]))
    file.close()