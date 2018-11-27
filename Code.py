import os
import random
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import pickle

#Set random seed for reproducable resutls
random.seed(0)

#Scrape webpage function
def scrap_data(url):
    html = requests.get(url).content
    return BeautifulSoup(html, 'html.parser')

def get_title_and_reviews(url):
    soup = scrap_data(url)
    title = soup.find(attrs={'data-hook': 'product-link'}).getText()
    total_page = soup.find_all('li', {'class':'page-button'}).pop().getText() if len(soup.find_all('li', {'class':'page-button'})) else 1
    reviews = []
    if total_page == 1:
        reviews = [review_data.getText() for review_data in soup.find_all(attrs={'data-hook' : 'review-body'})]
    else:
        for page in range(int(total_page)):
            next_soup = scrap_data(url + '&pageNumber=' + str(page + 1))
            next_reviews = [review_data.getText() for review_data in next_soup.find_all(attrs={'data-hook' : 'review-body'})]
            reviews = reviews + next_reviews
    return title,reviews

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
    words = nltk.word_tokenize(review) # Tokenize
    words = [word for word in words if not word in stop_words] # Remove stop words
    return words

#Load stopwords
stop_words = set(stopwords.words("english"))

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

#Split to pos and neg sets
pos_reviews = dataset[dataset["Recommended IND"] == 1]['Review Text']
neg_reviews = dataset[dataset["Recommended IND"] == 0]['Review Text']

# print("Positive words")
# draw_wordcloud(pos_reviews,'white')
# print("Negative words")
# draw_wordcloud(neg_reviews)

#Create pos and neg dic sets for training
pos_feats = [(word_feats(preprocessor(pos_review)), 'pos') for pos_review in pos_reviews]
neg_feats = [(word_feats(preprocessor(neg_review)), 'neg') for neg_review in pos_reviews]

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
# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
#
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

test = "This dress is amazing. It fits well and the color is pretty"
print(classifier.classify(word_feats(test.split())))

#scrape website
# reviews_url = "https://www.amazon.com/SWQZVT-Spaghetti-Sundress-Backless-Dresses/product-reviews/B07CG69L57/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
# title, reviews = get_title_and_reviews(reviews_url)
# for review in reviews:
#     tokens = nltk.word_tokenize(review)
