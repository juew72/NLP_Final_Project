import model
import scrape
import pickle

build_model = True

#Build model
if build_model:
    model.naivebayes()

#Load model
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

#Scrape data
reviews_url = "https://www.amazon.com/SWQZVT-Spaghetti-Sundress-Backless-Dresses/product-reviews/B07CG69L57/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
title, reviews = scrape.get_title_and_review(reviews_url)

#test model
for review in reviews:
    tag = classifier.classify(model.word_feats(model.preprocessor(review)))
    print("review: ", review)
    print("tag: ", tag)
