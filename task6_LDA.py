import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

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

doc_topic_df = pd.DataFrame(doc_topic_matrix)
print("Document-Topic Matrix:")
print(doc_topic_df)

# Create the Topic-Word matrix
topic_word_matrix = lda_model.get_topics()

topic_word_df = pd.DataFrame(topic_word_matrix, columns=[id2word[i] for i in range(len(id2word))])
print("Topic-Word Matrix:")
print(topic_word_df)