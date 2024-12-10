from flask import Flask, render_template, request, redirect, url_for
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app=Flask(__name__) 
model = joblib.load(open('./SVM_reviews.joblib','rb'))
tfidvectorizer = joblib.load('./tfidfvectorizer_reviews.joblib')
Stopwords_modified=set(stopwords.words('english')) - {'no', 'not','will', 'nor', 'but', 'however', 'although', 'yet', 'unfortunately', 'never', 'none', 'nobody', 'nowhere', 'nothing', 'neither', 'no one', 'without'}
CSV_FILE_PATH = './DATA.csv'
corpus = []
lemmatizer = WordNetLemmatizer()

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    url = request.form['Review']
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)

    review_list = scrape_amazon_reviews(url)
    overall_sentiment = ""  
    result = ""

    if review_list:
        data = pd.DataFrame(review_list)
        data.to_csv('./DATA.csv', index=False)
        data['Predicted_Sentiment'] = ""
        data = pd.read_csv('./DATA.csv')
        data = data.dropna(subset=['Reviews'])
        data==data.drop_duplicates()


        for i in range(data.shape[0]):
            cleaned_text = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['Reviews'])
            cleaned_text = re.sub(' +', ' ', cleaned_text)
            cleaned_text = cleaned_text.lower()
            tokenized_text = cleaned_text.split()  # Tokenization
            lemma_text = [lemmatizer.lemmatize(word) for word in tokenized_text if word not in Stopwords_modified]
            lemma_text_str = ' '.join(lemma_text)
            corpus.append(lemma_text_str)

        x_prediction = tfidvectorizer.transform(corpus)
        y_predictions = model.predict(x_prediction)
        y_predictions = list(map(sentiment_mapping, y_predictions))
        positive_count = sum(1 for sentiment in y_predictions if sentiment == 'positive')
        negative_count = sum(1 for sentiment in y_predictions if sentiment == 'negative')
        
        total_predictions = len(y_predictions)
        positive_percentage = (positive_count / total_predictions) * 100
        negative_percentage = (negative_count / total_predictions) * 100

        if positive_count > negative_count:
            overall_sentiment = f'{positive_percentage:.2f}% of the reviews are positive '
            result='The product is winning'
        else:
            overall_sentiment = f'{negative_percentage:.2f}% of the reviews are negative'
            result='The product is not winning'


    return render_template("index.html", prediction_text=overall_sentiment,winning=result, scroll_to_form=True)



def scrape_amazon_reviews(url):

    review_list = []
    def get_soup(url):
        splash_url = 'http://localhost:8050/render.html'
        params = {'url': url, 'wait': 2}
        splash_response = requests.get(splash_url, params=params)
        soup = BeautifulSoup(splash_response.text, 'html.parser')
        return soup

    def get_information(soup):
        reviews = soup.find_all('div', {'data-hook': 'review'})
        try:
            for review in reviews:
                a_tag = review.find('a', {'data-hook': 'review-title'})
                if a_tag:
                    span_tags = a_tag.find_all('span')
                    review_info = {}
                    for index, span_tag in enumerate(span_tags):
                        if index == 0:
                            rating = float(span_tag.text.replace('out of 5 stars', '').strip())
                            review_info['rate'] = rating
                    description = review.find('span', {'data-hook': 'review-body'})
                    if description:
                        review_info['Reviews'] = description.text.strip()
                        review_list.append(review_info)
        except:
            pass

    split_position = url.find('&pageNumber=')
    url_split=url[:split_position]
    
    for i in range(1, 5):
        soup = get_soup(f"{url_split}&pageNumber={i}")
        get_information(soup)
        print(f"Page {i}: {len(review_list)} reviews")
        if not soup.find('li', {'class': 'a-disabled a-last'}):
            pass
        else:
            print("No more pages left")
            break
    print(f"Total reviews scraped: {len(review_list)}")

    return review_list


def sentiment_mapping(x):
    if x==1:
        return "positive"
    else:
        return "negative"
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


