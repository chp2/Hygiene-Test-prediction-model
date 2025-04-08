import numpy as np
import nltk
import ast
import csv
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')

review_file = 'Hygiene/hygiene.dat'

f = open(review_file, 'r', encoding='utf-8')
review_list = f.readlines()

def preprocess_text(text):
    # Add any additional preprocessing steps here if needed
    return text.lower()

review_list = [preprocess_text(text) for text in review_list]

# Add your custom stopwords here
custom_stopwords = {'you', 'they', 'in', '$', '#', ';', '&', '.', 'i', '..', '...', ',','(',')', '-', '?', '!', ':', '*', "'s", "'ll", "n't", '``', "''", "'d"}
stopwords = STOPWORDS.union(custom_stopwords)

stemmer = PorterStemmer()

# Tokenize the review text, remove stopwords, and take stems
filtered_tokens = []
for review in review_list:
    token_list = word_tokenize(review)
    filtered_token_list = [word for word in token_list if word not in stopwords]
    stemmed_token_list = [stemmer.stem(word) for word in filtered_token_list]
    #print(stemmed_token_list)
    filtered_tokens.append(stemmed_token_list)

# Create a dictionary and corpus
id2word = corpora.Dictionary(filtered_tokens)
corpus = [id2word.doc2bow(text) for text in filtered_tokens]

# Train the LDA model
num_topics = 10
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, update_every=1, passes=10, alpha='auto', per_word_topics=True)

# Create the Document-Topic matrix
doc_topic_matrix = []

for doc_bow in corpus:
    doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
    doc_topic_matrix.append([topic_prob for topic_id, topic_prob in doc_topics])

array_doc_topic_matrix = np.array(doc_topic_matrix)

#doc_topic_df = pd.DataFrame(doc_topic_matrix)
#print("Document-Topic Matrix:")
#print(doc_topic_df)

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
#print(restaurant_categories_categories)

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

array_list_num_reviews = np.array(list_num_reviews)
#print(array_list_num_reviews.shape)
array_list_avg_rate= np.array(list_avg_rate)
#print(array_list_avg_rate.shape)
array_list_zipcode = np.array(list_zipcode)
array_category_existence_label = np.array(category_existence_label )
#print(array_category_existence_label.shape)



feature_table_LDA= np.column_stack((array_doc_topic_matrix, array_list_zipcode,array_list_num_reviews,array_list_avg_rate,array_category_existence_label))


with open('feature_table_LDA.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(feature_table_LDA)