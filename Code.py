import model
import scrape
import pickle

build_model = False

#Build model
if build_model:
    model.naivebayes()

#Load model
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

#Scrape data
reviews_url = "https://www.amazon.com/AUSELILY-Womens-Pockets-Pleated-T-Shirt/product-reviews/B07H5CWWYY/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
title, reviews = scrape.get_title_and_review(reviews_url)

#Classify reviews from amazon
pos_count = 0;
neg_count = 0
for review in reviews:
    tag = classifier.classify(model.word_feats(model.preprocessor(review)))
    if tag == 'pos':
        pos_count += 1
    else:
        neg_count += 1
    print("review: ", review)
    print("tag: ", tag)

print("Positive reviews:", pos_count)
print("Negative reviews:", neg_count)

#Calculate proportion. if percentage is pos_reviews are more than 70% => recommended
if pos_count/(pos_count + neg_count) >= 0.7:
    print(title, " is recommended")
else:
    print(title, " is not recommended")
