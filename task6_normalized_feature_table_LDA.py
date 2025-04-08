import csv

readfile = "feature_table_LDA.csv"

f = open(readfile, 'r', encoding='utf-8')
feature_list = []
feature_row = f.readline().strip()
while feature_row:
    feature_list.append(feature_row)
    feature_row = f.readline().strip()
    
zipcode_list = []
max_num_review = 0.0
max_rating = 5
for row in feature_list:
    row_list = row.split(",")
    zipcode_list.append(row_list[10])
    if float(row_list[11]) > max_num_review:
        max_num_review = float(row_list[11])

sorted_zipcode_list = list(sorted(set(zipcode_list)))


print("processing")
file = open('normalized_feature_table_LDA.csv', 'w', newline='')
writer = csv.writer(file)
for row in feature_list:
    row_list = row.split(",")
    #print(row_list)
    row_list[11] = float(row_list[11])/max_num_review
    row_list[12] = float(row_list[12])/max_rating
    zipcode_label = []
    for zipcode in sorted_zipcode_list:
            if row_list[10] == zipcode:
                zipcode_label.append(1)
            else:
                zipcode_label.append(0)
    row_list[10:11] = zipcode_label
    writer.writerow(row_list)