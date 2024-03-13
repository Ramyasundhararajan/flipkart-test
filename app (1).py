from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
from bs4 import BeautifulSoup
import csv
import selenium 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
app = Flask(__name__)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Function to read CSV file
def read_csv(file_path):
    reviews = []
    sentiments = []

    try:
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            for row in csv_reader:
                if len(row) == 2:
                    review_text, sentiment = row
                    reviews.append(review_text)
                    sentiments.append(sentiment)
                else:
                    print(f"Skipping row: {row}. It does not have two values.")
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return reviews, sentiments

train_file_path = 'amazon_review - Copy.csv'  # Change this to the path of your CSV file
train_reviews, train_sentiments = read_csv(train_file_path)

def get_flipkart_reviews(product_url):
    try:
        # Your existing code for sending a request
        response = requests.get(product_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Your existing code for extracting review elements
        review_elements = soup.find_all('div', class_='t-ZTKy')
        reviews = [element.text.strip() for element in review_elements]

        if not reviews:
            print("No reviews found. Checking for dynamic content.")

            all_reviews_url = soup.find_all('div', {'class': 'col JOpGWq'})[0]
            all_reviews_url = all_reviews_url.find_all('a')[-1]
            all_reviews_url = 'https://www.flipkart.com' + all_reviews_url.get('href')

            # Add code to fetch reviews using Selenium
            options = Options()
            options.add_argument('--headless')  # Run Chrome in headless mode (no GUI)
            driver = webdriver.Chrome(options=options)

            # Navigate to the URL with dynamic content
            driver.get(all_reviews_url)

            # Scroll to load dynamic content
            body = driver.find_element(By.TAG_NAME, 'body')
            for _ in range(5):  # Adjust the number of scrolls as needed
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(1)

            # Extract reviews after dynamic content loading
            dynamic_soup = BeautifulSoup(driver.page_source, 'html.parser')
            dynamic_review_elements = dynamic_soup.find_all('div', class_='t-ZTKy')
            dynamic_reviews = [element.text.strip() for element in dynamic_review_elements]

            # Combine reviews from static and dynamic content
            reviews += dynamic_reviews

            driver.quit()

        return reviews
    except Exception as e:
        return ["Error fetching reviews: " + str(e)]

@app.route('/predict_flipkart', methods=['POST'])
def predict_flipkart():
    if request.method == 'POST':
        flipkart_url = request.form.get('flipkart_url', '')

        if not flipkart_url:
            return "Flipkart URL is empty."

        flipkart_reviews = get_flipkart_reviews(flipkart_url)
        sentiments = [predict_sentiment(review) for review in flipkart_reviews]

        positive_reviews = [review for review, sentiment in zip(flipkart_reviews, sentiments) if sentiment == "Positive"]
        negative_reviews = [review for review, sentiment in zip(flipkart_reviews, sentiments) if sentiment == "Negative"]

        return render_template('results.html',
                               flipkart_url=flipkart_url,
                               positive_reviews=positive_reviews,
                               negative_reviews=negative_reviews)

def predict_sentiment(input_url):
    inputs = tokenizer(input_url, return_tensors="pt", max_length=64, padding=True, truncation=True)
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    _, predicted_class = torch.max(logits, dim=1)
    sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
