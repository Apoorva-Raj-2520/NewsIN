import requests
import re
import xml.etree.ElementTree as ET
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords


def fetch_article(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            title = soup.title.text
            
            # Extracting metadata
            metadata = {}
            metadata['url'] = url
            metadata['title'] = title
            
            # Find publisher date
            date_tag = soup.find('meta', {'property': 'article:published_time'})
            if date_tag:
                metadata['published_date'] = date_tag['content']
            else:
                metadata['published_date'] = None
            
            # Find author
            author_tag = soup.find('meta', {'name': 'author'})
            if author_tag:
                metadata['author'] = author_tag['content']
            else:
                metadata['author'] = None
            
            # Find image
            image_tag = soup.find('meta', {'property': 'og:image'})
            if image_tag:
                metadata['image'] = image_tag['content']
            else:
                metadata['image'] = None
            
            # Find content
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.text for p in paragraphs])
            
            return metadata, content
        else:
            return None, None
    except Exception as e:
        return None, None

def summarize_content(content, max_sentences=3):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
    # summary = ' '.join(sentences[:max_sentences])
    # return summary if len(summary) > 0 else None
    
    # Filter out stopwords
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    for word in nltk.word_tokenize(content.lower()):
        if word not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if len(sentence.split(' ')) < 30:  # Consider only reasonably short sentences
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
    
    # Get the most important sentences based on scores
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    summary = ' '.join(summarized_sentences)
    
    return summary if len(summary) > 0 else None

    


# Streamlit app
def run():
    st.title("News-IN: A Summarised News📰")
    image = Image.open('./Meta/newspaper.png')
    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")


    category = ['--Select--', 'Top News', 'Search Topic']
    cat_op = st.selectbox('Select your Category', category)

    if cat_op == category[0]:
        st.warning('Please select Type!!')
    elif cat_op == category[1]:
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        st.subheader("Click to load the Top News")
        if st.button("Load"):
            try:
                # Sample URL for searching articles using Google RSS
                search_url = f"https://news.google.com/rss/"
                response = requests.get(search_url)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    count = 0
                    for item in root.iter('item'):
                        if count >= no_of_news:  # Display only top 5 articles
                            break
                        title = item.find('title').text
                        url = item.find('link').text
                        metadata, content = fetch_article(url)
                        if metadata and content:
                            summary = summarize_content(content)
                            if summary:
                                count += 1
                                st.write(f"Title: {metadata['title']}")
                                st.write(f"URL: {metadata['url']}")
                                st.write(f"Published Date: {metadata['published_date']}")
                                st.write(f"Author: {metadata['author']}")
                                if metadata['image']:
                                    st.image(metadata['image'], caption='Article Image', use_column_width=True)
                                else:
                                    st.write("No image available")
                                st.write("Content:")
                                st.write(content)
                                st.write("Summary:")
                                st.write(summary)
                                st.write("\n")
                        else:
                            continue
                else:
                    st.error("Failed to search articles.")
            except Exception as e:
                st.error("Error searching articles: {}".format(e))

    elif cat_op == category[2]:
        keyword = st.text_input("Enter your Topic🔍")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        if st.button("Search"):
            if keyword:
                try:
                    # Sample URL for searching articles using Google RSS
                    search_url = f"https://news.google.com/rss/search?q={keyword}"
                    response = requests.get(search_url)
                    if response.status_code == 200:
                        root = ET.fromstring(response.content)
                        count = 0
                        for item in root.iter('item'):
                            if count >= no_of_news:  # Display only top 5 articles
                                break
                            title = item.find('title').text
                            url = item.find('link').text
                            metadata, content = fetch_article(url)
                            if metadata and content:
                                summary = summarize_content(content)
                                if summary:
                                    count += 1
                                    st.write(f"Title: {metadata['title']}")
                                    st.write(f"URL: {metadata['url']}")
                                    st.write(f"Published Date: {metadata['published_date']}")
                                    st.write(f"Author: {metadata['author']}")
                                    if metadata['image']:
                                        st.image(metadata['image'], caption='Article Image', use_column_width=True)
                                    else:
                                        st.write("No image available")
                                    st.write("Content:")
                                    st.write(content)
                                    st.write("Summary:")
                                    st.write(summary)
                                    st.write("\n")
                            else:
                                continue
                    else:
                        st.error("Failed to search articles.")
                except Exception as e:
                    st.error("Error searching articles: {}".format(e))
            else:
                st.warning("Please enter a keyword to search articles.")
run()