import os
import pandas
import nltk
from bs4 import BeautifulSoup
import requests
import re

#Scrape webpage function
def scrap_data(url):
    html = requests.get(url).content
    return BeautifulSoup(html, 'html.parser')


#Get path for dataset
current_working_directory = os.getcwd()
dataset_path = current_working_directory + '/Dataset/Womens Clothing E-Commerce Reviews.csv'


#Load dataset
df = pandas.read_csv(dataset_path)

#Get reviews(x) and recommended(y) for model
reviews = df["Review Text"]
recommended = df["Recommended IND"]


#scrape website
reviews_url = "https://www.amazon.com/AUSELILY-Womens-Pockets-Pleated-T-Shirt/product-reviews/B07H5CWWYY/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
soup = scrap_data(reviews_url)
title = soup.find(attrs={"data-hook": "product-link"}).getText()
reviews = [review_data.getText() for review_data in soup.find_all(attrs={"data-hook" : "review-body"})]
for review in reviews:
    tokens = nltk.word_tokenize(review)
