import requests
import re
import xml.etree.ElementTree as ET
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup

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
    summary = ' '.join(sentences[:max_sentences])
    return summary if len(summary) > 0 else None

# Streamlit app
def run():
    st.title("InNews🇮🇳: A Summarised News📰")
    image = Image.open('./Meta/newspaper.png')
    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")
        
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