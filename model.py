import os
import random
import pandas as pd
from wordcloud import WordCloud,STOPWORDS
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import re
import pickle

#Set random seed for reproducable resutls
random.seed(0)

#Load stopwords
stop_words = set(stopwords.words("english"))

#Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#draw worcloud
def draw_wordcloud(reviews, color='black'):
    data = [str(review) for review in reviews]
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                             if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'
                             ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000
                          ).generate(cleaned_word)
    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

# Create dictionary object for nlkt NaivesBayesian
def word_feats(words):
    return dict([(word, True) for word in words])

# Preprocess reviews
def preprocessor(review) :
    cleaned_review = re.sub('[^a-zA-Z]', ' ', str(review))
    cleaned_review = cleaned_review.lower()
    words = nltk.word_tokenize(cleaned_review) # Tokenization
    words = [word for word in words if not word in stop_words] # Remove stop words
    words = [wordnet_lemmatizer.lemmatize(word) for word in words] #Lemmatize words
    return words

def naivebayes():
    #Get path for dataset
    current_working_directory = os.getcwd()
    dataset_path = current_working_directory + '/Dataset/Womens Clothing E-Commerce Reviews.csv'

    #Load dataset
    dataset = pd.read_csv(dataset_path)[['Title', 'Review Text','Recommended IND']]

    #Replace missing review text with title when missing
    for index, row in dataset.iterrows():
        review_text = row['Review Text']
        if pd.isnull(review_text):
            row['Review Text'] = row['Title']

    #Drop the rest of na values
    dataset = dataset.dropna();

    #Basic information
    print('Number of review text in the dataset grouped by: {}.'.format(dataset.groupby('Recommended IND')['Review Text'].count()))

    #Split to pos and neg sets
    pos_reviews = dataset[dataset["Recommended IND"] == 1]['Review Text']
    neg_reviews = dataset[dataset["Recommended IND"] == 0]['Review Text']

    # print("Positive words")
    draw_wordcloud(pos_reviews,'white')
    print("Negative words")
    draw_wordcloud(neg_reviews)

    #Create pos and neg dic sets for training
    pos_feats = [(word_feats(preprocessor(pos_review)), 'pos') for pos_review in pos_reviews]
    neg_feats = [(word_feats(preprocessor(neg_review)), 'neg') for neg_review in neg_reviews]

    #Get train and test set
    pos_train, pos_test = train_test_split(pos_feats,test_size = 0.3)
    neg_train, neg_test = train_test_split(neg_feats,test_size = 0.3)

    train_feats = pos_train + neg_train
    test_feats = pos_test + neg_test

    #Train model
    print('train on %d instances, test on %d instances' % (len(train_feats), len(test_feats)))
    classifier = NaiveBayesClassifier.train(train_feats)

    #Test model
    print('accuracy:', nltk.classify.util.accuracy(classifier, test_feats))

    #Most informative features
    classifier.show_most_informative_features()

    #save classifier for testing
    save_classifier = open("naivebayes.pickle","wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

