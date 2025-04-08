import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
import ast
import numpy as np
import csv

nltk.download('punkt')

print('loading additional')
review_file = 'Hygiene/hygiene.dat.additional'
f = open(review_file, 'r', encoding='utf-8')
additional = f.readlines()

all_categories = set()
list_num_reviews = []
list_avg_rate = []
list_zipcode = []
restaurant_categories = []
for attributes in additional:
    categories_str, rest_str = attributes.split('",', 1)
    categories_str = categories_str[1:]
    categories_list = ast.literal_eval(categories_str)
    categories_list.remove('Restaurants')
    restaurant_categories.append(categories_list)
    numbers_str = rest_str.split(',')
    zip_code = int(numbers_str[0])
    num_reviews = int(numbers_str[1])
    avg_rate = float(numbers_str[2])
    all_categories.update(categories_list)
    list_zipcode.append(zip_code)
    list_num_reviews.append(num_reviews)
    list_avg_rate.append(avg_rate)

all_categories_list = list(sorted(all_categories))

category_existence_label = []
for categories in restaurant_categories:
    label = []
    for category in all_categories_list:
        if category in categories:
            label.append(1)
        else:
            label.append(0)
    category_existence_label.append(label)
#array_list_num_reviews = np.array(list_num_reviews)
#array_list_avg_rate= np.array(list_avg_rate)
#array_list_zipcode = np.array(list_zipcode)
#array_category_existence_label = np.array(category_existence_label )

print('end of additional label')
########################################################################

print('TF_IDF processing')
review_TF_IDF_file = 'review_TF_IDF.csv'
f = open(review_TF_IDF_file, 'r', encoding='utf-8')
file = open('feature_table_TF_IDF.csv', 'w', newline='')
writer = csv.writer(file)
f.readline().strip() #read column name
tf_idf_str = f.readline().strip()
for i in range(len(list_zipcode)):
    tf_idf_list = tf_idf_str.split(",")
    tf_idf_list.append(list_zipcode[i])
    tf_idf_list.append(list_num_reviews[i])
    tf_idf_list.append(list_avg_rate[i])
    tf_idf_list.extend(category_existence_label[i])
    writer.writerow(tf_idf_list)
    tf_idf_str = f.readline().strip()