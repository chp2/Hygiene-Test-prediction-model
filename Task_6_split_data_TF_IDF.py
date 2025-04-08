import csv

readfile = "normalized_feature_table_TF_IDF.csv"
f = open(readfile, 'r', encoding='utf-8')
file = open('X_train_TF_IDF.csv', 'w', newline='')
for i in range(546):
    feature_row = f.readline()
    file.write(feature_row)
file.close()

file = open('X_test_TF_IDF.csv', 'w', newline='')
for i in range(12753):
    feature_row = f.readline()
    file.write(feature_row)
file.close()

'''
readfile = "Hygiene/hygiene.dat.labels"
f = open(readfile, 'r', encoding='utf-8')
file = open('Y_train.csv', 'w', newline='')
for i in range(546):
    feature_row = f.readline()
    file.write(feature_row)
'''