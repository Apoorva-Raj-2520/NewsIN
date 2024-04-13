# News-IN: A Summarised News 📰
News-IN is a Streamlit app that fetches and summarizes news articles from various sources based on user preferences. This app leverages web scraping and natural language processing techniques to provide concise summaries of news articles.

## Features

### Top News: 
View summarized top news articles fetched from Google RSS feeds.
### Search Topic: 
Search and summarize news articles based on user-specified keywords.

## Technologies Used
Python
Streamlit
Requests
Beautiful Soup (bs4)
NLTK (Natural Language Toolkit)
Pillow (PIL) for image display

# Installation
### Clone the repository:
git clone https://github.com/Apoorva-Raj-2520/NewsIN.git
cd newsin

### Install the required Python packages:
pip install -r requirements.txt

# Usage
### Run the Streamlit app:
streamlit run app.py
#### On VS Code:
python3 -m streamlit run app.py

### Open the Streamlit app in your web browser (by default, Streamlit runs on http://localhost:8501):
http://localhost:8501

### Select the desired category (Top News or Search Topic) and follow the prompts to view and summarize news articles.