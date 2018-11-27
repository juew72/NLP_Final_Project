import requests
from bs4 import BeautifulSoup

#Scrape webpage function
def scrap_data(url):
    html = requests.get(url).content
    return BeautifulSoup(html, 'html.parser')

def get_title_and_review(url):
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
