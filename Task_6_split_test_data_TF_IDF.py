readfile = "X_test_TF_IDF.csv"
f = open(readfile, 'r', encoding='utf-8')

for i in range(26):
    filename = 'X_test_TF_IDF' + str(i) + '.csv'
    file = open(filename, 'w', newline='')
    for j in range(500):
        feature_row = f.readline()
        file.write(feature_row)
    file.close()