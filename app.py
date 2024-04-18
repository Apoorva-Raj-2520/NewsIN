import requests
import re
import xml.etree.ElementTree as ET
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup
import math

import nltk
from nltk.corpus import stopwords

from collections import Counter


ideal = 20.0
stop_words = set()
stop_words_2 = set()

summ = {}
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
            # paragraphs = soup.find_all('p')
            # content = '\n'.join([p.text for p in paragraphs])
            
            paragraphs = soup.select('p')
            if paragraphs:
                content = '\n'.join([p.get_text(separator='\n', strip=True) for p in paragraphs])
            else:
                content = ''
            return metadata, content
        else:
            return None, None
    except Exception as e:
        return None, None
#################

def split_words(text):
    """Split a string into array of words
    """
    try:
        text = re.sub(r'[^\w ]', '', text)  # strip special chars
        return [x.strip('.').lower() for x in text.split()]
    except TypeError:
        return None


def keywords(text):
    """Get the top 10 keywords and their frequency scores ignores blacklisted
    words in stop_words, counts the number of occurrences of each word, and
    sorts them in reverse natural order (so descending) by number of
    occurrences.
    """
    NUM_KEYWORDS = 10
    text = split_words(text)
    # of words before removing blacklist words
    if text:
        num_words = len(text)
        text = [x for x in text if x not in stop_words]
        freq = {}
        for word in text:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

        min_size = min(NUM_KEYWORDS, len(freq))
        keywords = sorted(freq.items(),
                          key=lambda x: (x[1], x[0]),
                          reverse=True)
        keywords = keywords[:min_size]
        keywords = dict((x, y) for x, y in keywords)

        for k in keywords:
            articleScore = keywords[k] * 1.0 / max(num_words, 1)
            keywords[k] = articleScore * 1.5 + 1
        return dict(keywords)
    else:
        return dict()


def keywords_2(text):
    """Get the top 10 keywords and their frequency scores ignores blacklisted
    words in stop_words, counts the number of occurrences of each word, and
    sorts them in reverse natural order (so descending) by number of
    occurrences.
    """
    NUM_KEYWORDS = 10
    text = split_words(text)
    # of words before removing blacklist words
    if text:
        num_words = len(text)
        text = [x for x in text if x not in stop_words_2]
        freq = {}
        for word in text:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

        min_size = min(NUM_KEYWORDS, len(freq))
        keywords = sorted(freq.items(),
                          key=lambda x: (x[1], x[0]),
                          reverse=True)
        keywords = keywords[:min_size]
        keywords = dict((x, y) for x, y in keywords)

        for k in keywords:
            articleScore = keywords[k] * 1.0 / max(num_words, 1)
            keywords[k] = articleScore * 1.5 + 1
        return dict(keywords)
    else:
        return dict()

def split_sentences(text):
    """Split a large string into sentences
    """
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)
    sentences = [x.replace('\n', '') for x in sentences if len(x) > 10]
    return sentences
  

def summarize_content(content, title, max_sents=5):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
    
    # Filter out stopwords
    global stop_words 
    global stop_words_2
    stop_words = set(stopwords.words('english'))
    
    if not content or not title or max_sents <= 0:
        return []

    summaries = []
    summaries_2 =[]
    # sentences = split_sentences(content)
    keys = keywords(content)
    keys_2 = keywords_2(content)
    titleWords = split_words(title)

    # Score sentences, and use the top 5 or max_sents sentences
    ranks = score(sentences, titleWords, keys).most_common(max_sents)
    for rank in ranks:
        summaries.append(rank[0])
    summaries.sort(key=lambda summary: summary[0])
    summary_comb = ""
    for summary in summaries:
        summary_comb += summary[1]
    # return [summary[1] for summary in summaries]
    global summ
    summ[0] = summary_comb

    # with custom keywords
    ranks_2 = score(sentences, titleWords, keys_2).most_common(max_sents)
    for rank in ranks_2:
        summaries_2.append(rank[0])
    summaries_2.sort(key=lambda summary: summary[0])
    summary_comb = ""
    for summary in summaries_2:
        summary_comb += summary[1]
    # return [summary[1] for summary in summaries]
    summ[2] = summary_comb


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
                if len(sentence.split(' ')) < 50:  # Consider only reasonably short sentences
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
    
    # Get the most important sentences based on scores
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
    summary = ' '.join(summarized_sentences)
    
    summ[1] = summary
    sum_final=""
    for i in range(len(summ)):
        sum_final+="{} :\n".format(i+1)
        sum_final+=summ[i]
        sum_final+="\n \n"
    return sum_final if len(sum_final) > 0 else None

    
#scoring
def score(sentences, titleWords, keywords):
    """Score sentences based on different features
    """
    senSize = len(sentences)
    ranks = Counter()
    for i, s in enumerate(sentences):
        sentence = split_words(s)
        titleFeature = title_score(titleWords, sentence)
        sentenceLength = length_score(len(sentence))
        sentencePosition = sentence_position(i + 1, senSize)
        sbsFeature = sbs(sentence, keywords)
        dbsFeature = dbs(sentence, keywords)
        frequency = (sbsFeature + dbsFeature) / 2.0 * 10.0
        # Weighted average of scores from four categories
        totalScore = (titleFeature*1.5 + frequency*2.0 +
                      sentenceLength*1.0 + sentencePosition*1.0)/4.0
        ranks[(i, s)] = totalScore
    return ranks


def sbs(words, keywords):
    score = 0.0
    if (len(words) == 0):
        return 0
    for word in words:
        if word in keywords:
            score += keywords[word]
    return (1.0 / math.fabs(len(words)) * score) / 10.0


def dbs(words, keywords):
    if (len(words) == 0):
        return 0
    summ = 0
    first = []
    second = []

    for i, word in enumerate(words):
        if word in keywords:
            score = keywords[word]
            if first == []:
                first = [i, score]
            else:
                second = first
                first = [i, score]
                dif = first[0] - second[0]
                summ += (first[1] * second[1]) / (dif ** 2)
    # Number of intersections
    k = len(set(keywords.keys()).intersection(set(words))) + 1
    return (1 / (k * (k + 1.0)) * summ)


def length_score(sentence_len):
    return 1 - math.fabs(ideal - sentence_len) / ideal


def title_score(title, sentence):
    if title:
        title = [x for x in title if x not in stop_words]
        count = 0.0
        for word in sentence:
            if (word not in stop_words and word in title):
                count += 1.0
        return count / max(len(title), 1)
    else:
        return 0


def sentence_position(i, size):
    """Different sentence positions indicate different
    probability of being an important sentence.
    """
    normalized = i * 1.0 / size
    if (normalized > 1.0):
        return 0
    elif (normalized > 0.9):
        return 0.15
    elif (normalized > 0.8):
        return 0.04
    elif (normalized > 0.7):
        return 0.04
    elif (normalized > 0.6):
        return 0.06
    elif (normalized > 0.5):
        return 0.04
    elif (normalized > 0.4):
        return 0.05
    elif (normalized > 0.3):
        return 0.08
    elif (normalized > 0.2):
        return 0.14
    elif (normalized > 0.1):
        return 0.23
    elif (normalized > 0):
        return 0.17
    else:
        return 0


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
                            summary = summarize_content(content, metadata['title'])
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
                                summary = summarize_content(content, metadata['title'])
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