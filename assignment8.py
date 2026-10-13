
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud

# Download required NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))


# ==================== Q1 ====================

print("\n========== Q1: TEXT PREPROCESSING ==========")

text = """
Artificial intelligence is transforming the world through technology.
Machine learning helps computers learn from data and improve performance.
AI is used in healthcare, education, and transportation.
It can automate repetitive tasks and solve complex problems.
Responsible AI development is important for a better future.
Technology should help people live safer and smarter lives.
"""

lower_text = text.lower()
clean_text = re.sub(r'[^\w\s]', '', lower_text)

print("Lowercase text:\n", lower_text)
print("\nText without punctuation:\n", clean_text)

sentences = sent_tokenize(text)
words_split = clean_text.split()
words_nltk = word_tokenize(clean_text)

print("\nSentences:", sentences)
print("\nPython split():", words_split)
print("\nNLTK word_tokenize():", words_nltk)

print("\nComparison:")
print("split() separates text using whitespace.")
print("word_tokenize() handles linguistic tokenization.")
print("Punctuation is already removed here, so the results are similar.")

filtered_words = [
    word for word in words_nltk
    if word.isalpha() and word not in stop_words
]

print("\nWords after stopword removal:", filtered_words)

frequency = Counter(filtered_words)
print("\nWord Frequency Distribution:")
print(frequency)


# ==================== Q2 ====================

print("\n========== Q2: STEMMING AND LEMMATIZATION ==========")

alphabet_words = re.findall(r'\b[a-zA-Z]+\b', lower_text)

print("Alphabet-only words:", alphabet_words)

filtered_q2 = [
    word.lower() for word in alphabet_words
    if word.lower() not in stop_words
]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [
    stemmer.stem(word) for word in filtered_q2
]

lemmatized_words = [
    lemmatizer.lemmatize(word) for word in filtered_q2
]

print("\nAfter stopword removal:", filtered_q2)
print("\nStemmed words:", stemmed_words)
print("\nLemmatized words:", lemmatized_words)

print("\nComparison:")
print("Stemming removes word endings using rules.")
print("Lemmatization returns dictionary forms of words.")
print("Stemming is useful for quick text matching.")
print("Lemmatization is useful when meaningful words matter.")


# ==================== Q3 ====================

print("\n========== Q3: BOW AND TF-IDF ==========")

documents = [
    "Artificial intelligence improves healthcare and education.",
    "Electric cars reduce pollution and protect the environment.",
    "Online education makes learning accessible and flexible."
]

# Bag of Words
count_vectorizer = CountVectorizer(stop_words='english')
bow_matrix = count_vectorizer.fit_transform(documents)

print("\nBag of Words vocabulary:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words matrix:")
print(bow_matrix.toarray())

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF matrix:")
print(np.round(tfidf_matrix.toarray(), 3))

for i, document in enumerate(documents):
    scores = tfidf_matrix[i].toarray().flatten()
    top_indices = scores.argsort()[::-1][:3]

    print(f"\nText {i + 1}:", document)
    print("Top 3 TF-IDF keywords:")

    for index in top_indices:
        print(feature_names[index], round(scores[index], 3))

print("\nInterpretation:")
print("Bag of Words counts word occurrences.")
print("TF-IDF gives higher importance to relatively distinctive words.")


# ==================== Q4 ====================

print("\n========== Q4: TEXT SIMILARITY ==========")

text1 = """
Artificial intelligence allows computers to learn from data.
AI supports automation and intelligent decision making.
Machine learning is an important part of AI.
It is used in healthcare and education.
AI systems can improve productivity.
"""

text2 = """
Blockchain stores information in linked digital blocks.
It supports secure and transparent transactions.
Cryptocurrency commonly uses blockchain technology.
Blockchain records can be difficult to modify.
This technology is useful for digital security.
"""

# Preprocessing and tokenization
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

tokens1 = preprocess(text1)
tokens2 = preprocess(text2)

print("AI tokens:", tokens1)
print("Blockchain tokens:", tokens2)

# Jaccard Similarity
set1 = set(tokens1)
set2 = set(tokens2)

jaccard = len(set1 & set2) / len(set1 | set2)

print("\nJaccard Similarity:", round(jaccard, 4))

# Cosine Similarity using TF-IDF
similarity_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_pair = similarity_vectorizer.fit_transform([text1, text2])

cosine = cosine_similarity(tfidf_pair[0], tfidf_pair[1])[0][0]

print("Cosine Similarity:", round(cosine, 4))

print("\nAnalysis:")
print("Jaccard measures the overlap between unique words.")
print("Cosine similarity compares TF-IDF weighted word vectors.")
print("Cosine similarity considers word importance, while")
print("Jaccard is simpler and focuses only on shared vocabulary.")


# ==================== Q5 ====================

print("\n========== Q5: SENTIMENT ANALYSIS ==========")

reviews = [
    "This product is amazing and works perfectly.",
    "I love the quality and the excellent customer service.",
    "The product is terrible and completely disappointing.",
    "The delivery was slow and the item was damaged.",
    "The product arrived today and is working.",
    "The service is okay but nothing special."
]

positive_reviews = []

for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
        positive_reviews.append(review)
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print("\nReview:", review)
    print("Polarity:", round(polarity, 3))
    print("Subjectivity:", round(subjectivity, 3))
    print("Sentiment:", sentiment)

# Word Cloud for positive reviews
print("\nPositive reviews:", positive_reviews)

if positive_reviews:
    positive_text = " ".join(positive_reviews)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words
    ).generate(positive_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Positive Reviews")
    plt.show()


# ==================== Q6 ====================

print("\n========== Q6: LSTM TEXT GENERATION ==========")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

training_text = """
technology changes the world every day
artificial intelligence helps people solve problems
machine learning allows computers to discover patterns
students learn programming to build useful applications
good software makes difficult tasks easier
data helps computers understand information
creative thinking leads to innovative technology
computer science creates opportunities for the future
learning new skills improves our knowledge
technology connects people across the world
artificial intelligence can support education
students use computers to develop creative projects
programming teaches logical and analytical thinking
machine learning can recognize patterns in data
innovation helps society progress every day
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([training_text])

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for line in training_text.strip().split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding='pre'
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = np.eye(total_words)[y]

model = Sequential([
    Embedding(total_words, 16),
    LSTM(64),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X, y, epochs=200, verbose=0)

print("\nLSTM training completed!")

def generate_text(seed_text, next_words=8):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predictions = model.predict(token_list, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))

        if predicted_index == 0:
            break

        output_word = tokenizer.index_word.get(predicted_index, '')

        if not output_word:
            break

        seed_text += " " + output_word

    return seed_text

seed_word = "technology"

print("\nGenerated text:")
print(generate_text(seed_word, 8))
print(generate_text("artificial", 8))
print(generate_text("students", 8))
