readfile = "X_test_TF_IDF.csv"
f = open(readfile, 'r', encoding='utf-8')
print(len(f.readline().strip().split(',')))
