import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

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

#Get path for dataset
current_working_directory = os.getcwd()
dataset_path = current_working_directory + '/Dataset/Womens Clothing E-Commerce Reviews.csv'


#Load dataset
dataset = pd.read_csv(dataset_path)[['Review Text','Recommended IND']]

stop_words = set(stopwords.words("english"))

#Get train and test set
train, test = train_test_split(dataset,test_size = 0.3)

#Split data into positive and negative
train_pos = train[train['Recommended IND'] == 1]
train_pos = train_pos['Review Text']
train_neg = train[ train['Recommended IND'] == 0]
train_neg = train_neg['Review Text']

print("Positive words")
draw_wordcloud(train_pos,'white')
print("Negative words")
draw_wordcloud(train_neg)

#Processed reviews in train dataset
processed_reviews_train = []
for index, row in train.iterrows():
    review = row['Review Text']
    review_words = nltk.word_tokenize(review)
    review_words = [word for word in review_token if not word in stop_words]
    processed_reviews_train.append((review_words, row['Recommend IND']))


# Extracting word features
def get_word_list(reviews):
    wordlist = []
    for review, sentiment in reviews:
        wordlist.append()
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#scrape website
# reviews_url = "https://www.amazon.com/SWQZVT-Spaghetti-Sundress-Backless-Dresses/product-reviews/B07CG69L57/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
# title, reviews = get_title_and_reviews(reviews_url)
# for review in reviews:
#     tokens = nltk.word_tokenize(review)
