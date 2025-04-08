import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#import dask.dataframe as dd


column_names = ['c'+str(name) for name in range(96305)]

# Load the dataset from a CSV file
print('x_train loading')
X_train = pd.read_csv('X_train_TF_IDF.csv', header=None, names=column_names)
print('y_train loading')
y_train = pd.read_csv('Y_train.csv', header=None, names=['target'])
#y_train = np.array(y_train)
y_train = y_train.values.ravel()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#print('transform')
#X_test = scaler.transform(X_test)
log_reg = LogisticRegression(max_iter=200)

# Train the model on the training data
log_reg.fit(X_train, y_train)

print('x_test loading')
for i in range(26):
    filename = 'X_test_TF_IDF' + str(i) + '.csv'
    X_test = pd.read_csv(filename, header=None, names=column_names)
    #X_test =  pd.read_csv(filename, header=None)
    print(i)
    y_pred = log_reg.predict(X_test)

    y_pred_list = [[pred] for pred in y_pred]
    y_filename = 'Log_reg_Y_pred_TF_IDF' + str(i) + '.csv'
    file = open(y_filename, 'w', newline='')
    file.write(str(y_pred[0]))
    for i in range(1, len(y_pred_list)):
        #print(i)
        file.write('\n')
        file.write(str(y_pred[i]))
    file.close()

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")

#print("Classification Report:")
#print(classification_report(y_test, y_pred))