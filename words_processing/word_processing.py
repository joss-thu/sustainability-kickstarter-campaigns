import sklearn.feature_extraction.text as text
import re
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

#Easy test: tf-idf
# tf-idf 
#STOP_WORDS='english', exclude also numbers
STOP_WORDS = list(text.ENGLISH_STOP_WORDS.union([str(i) for i in range(10)]))
MIN_DOCS= .05
TOKEN_PATTERN= '(?u)\\b[a-zA-Z]{2,}\\b'

def stem(extracts_list):
    stemmer = SnowballStemmer("english")
    return [' '.join([stemmer.stem(word.lower()) for word in word_tokenize(extract)]) for extract in extracts_list]

def extract_to_stemmed_list(extract):
    extract= re.sub(r'[\W,]+', ' ', extract).replace('-', ' ').lower().split()
    return stem(extract)

def get_keywords_count(df, ranked_words, count_column, column_ranked_words='ranked_words', threshold=0.05):
    #Returns the detected keywords and its count in a text corpus. The keywords are words from ranked_words list above a certain defined threshold
    df[count_column] = 0
    df['combined_description']=''
    
    df.loc[:,'combined_description'] = df.loc[:,'campaign_name'] + ' ' + df.loc[:,'blurb']

    #Get list of words appearing in the ranked_words and greater than the defined threshold
    df.loc[:,column_ranked_words] = df.loc[:,'combined_description'].apply(
        lambda text_description: [word if word in ranked_words.index and ranked_words.loc[word] > threshold else '' 
                        for word in extract_to_stemmed_list(text_description)]
                        ).apply(lambda word: list(filter(None, word)))
    
    #Get count of ranked words for each row
    df.loc[:,count_column]= df.loc[:,column_ranked_words].apply(len)
    #Drop the combined description
    df.drop(columns=['combined_description'], axis=1,inplace=True)
    return df

def get_word_count_in_classified_blurbs(df, count_column):
    #Find the prescence of keywords in rows with already manually classified descriptions
    return df[count_column].apply(lambda x: 'No keyword' if x == 0 else 'at least one keyword').value_counts()

#Function to get a list of top ranked words in a literature corpus using tf-idf algorithm
def get_ranked_words(vocabulary:list,text_extracts:list, stop_words=STOP_WORDS, min_df= MIN_DOCS, token_pattern=TOKEN_PATTERN, *args, **kwargs):
    #tf_idf_model
    tf_idf_model = TfidfVectorizer(stop_words=stop_words, min_df= min_df, token_pattern=token_pattern)
    #stem extracts list
    vocabulary = stem([text for text in vocabulary])
    #Vectorization of corpus
    tf_idf_vector = tf_idf_model.fit(vocabulary)
    tf_idf_vector = tf_idf_model.transform(text_extracts)
    # #Get original terms in the corpus
    words_set = tf_idf_model.get_feature_names_out()
    # #Data frame to show the TF-IDF scores of each document
    df_tf_idf = pd.DataFrame.sparse.from_spmatrix(tf_idf_vector, columns=words_set)
    # Calculate the sum of TF-IDF scores for each word
    word_importance = df_tf_idf.mean(axis=0)
    # Sort words based on the sum of TF-IDF scores
    ranked_words = word_importance.sort_values(ascending=False)
    return ranked_words
