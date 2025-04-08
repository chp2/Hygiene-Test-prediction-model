import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize

#### get list of review ###
# Directory containing the text files
review_file = 'Hygiene/hygiene.dat'

f = open(review_file, 'r', encoding='utf-8')
review_list = f.readlines()

#Preprocess text data (basic preprocessing)
def preprocess_text(text):
    # Add any additional preprocessing steps here if needed
    return text.lower()

review_list = [preprocess_text(text) for text in review_list]

# define stop words
custom_stopwords = {'you', 'is', 'another', 'but',  'they', 'in', '$', '#', ';', '&', '.', 'i', '..', '...', ',','(',')'}  # Add your custom stopwords here
stopwords = STOPWORDS.union(custom_stopwords)

# Tokenize the review text and remove stopwords
filtered_tokens = []
for review in review_list:
    filtered_tokens.append(word for word in word_tokenize(review) if word not in stopwords)

# Join tokens to form sentences/documents
documents = [' '.join(tokens) for tokens in filtered_tokens]

# Construct tf_idf array
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_array = tfidf_matrix.toarray()

#convert tf_idf array into data frame
feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(tfidf_array, columns=feature_names)

#save dataframe to csv
df.to_csv('review_TF_IDF.csv', index=False)

# Optionally, read the CSV file back to confirm
#df_loaded = pd.read_csv('data.csv')
