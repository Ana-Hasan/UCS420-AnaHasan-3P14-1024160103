
# ================================================================
# UCS420: COGNITIVE COMPUTING
# ASSIGNMENT 7 - NLP-I
# Complete solution for Questions 1-6
# ================================================================

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from itertools import islice

# Download required NLTK resources
for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger_eng"
]:
    nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    RegexpStemmer,
    WordNetLemmatizer
)

print("=" * 75)
print("UCS420 COGNITIVE COMPUTING - ASSIGNMENT 7: NLP-I")
print("=" * 75)


# ================================================================
# IMPORTANT: REUSE THE EXISTING STARTER NOTEBOOK DATASET
# ================================================================

# The starter notebook must already have created the dataset.
# We do NOT call generate_dataset() or replace its dataset.

candidate_names = [
    "tickets",
    "support_tickets",
    "ticket_data",
    "dataset",
    "data",
    "generated_tickets"
]

ticket_data = None
dataset_variable = None

for name in candidate_names:
    obj = globals().get(name)

    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        ticket_data = obj
        dataset_variable = name
        break

    if isinstance(obj, pd.DataFrame) and len(obj) > 0:
        ticket_data = obj.to_dict("records")
        dataset_variable = name
        break

if ticket_data is None:
    raise NameError(
        "Could not automatically find the existing ticket dataset. "
        "Set ticket_data = YOUR_EXISTING_DATASET using the variable "
        "name from the starter notebook, then rerun this cell."
    )

print(f"\nUsing existing dataset variable: {dataset_variable}")
print("Number of tickets:", len(ticket_data))


# Extract ticket text without changing the original dataset.
def get_ticket_text(ticket):
    if isinstance(ticket, str):
        return ticket

    if isinstance(ticket, dict):
        preferred_fields = [
            "text", "description", "body", "message",
            "ticket", "content", "issue", "subject"
        ]

        for field in preferred_fields:
            if field in ticket and isinstance(ticket[field], str):
                return ticket[field]

        # Fallback: combine textual fields
        text_fields = [
            str(value) for value in ticket.values()
            if isinstance(value, str)
        ]
        if text_fields:
            return " ".join(text_fields)

    return str(ticket)


raw_tickets = [get_ticket_text(ticket) for ticket in ticket_data]


# ================================================================
# Q1. PREPROCESSING
# ================================================================

print("\n" + "=" * 75)
print("Q1: PREPROCESSING")
print("=" * 75)

# Required existing starter function
if "process_ticket" not in globals():
    raise NameError(
        "process_ticket() is missing. Run the starter notebook cells first."
    )

# Run process_ticket() on every ticket.
processed_tickets = [
    process_ticket(ticket) for ticket in ticket_data
]

stop_words = set(stopwords.words("english"))

# Token-level punctuation detection
def is_punctuation_token(token):
    return bool(token) and all(
        char in string.punctuation for char in token
    )


def basic_tokenize(text):
    return word_tokenize(text)


def get_final_tokens(text):
    tokens = basic_tokenize(text)

    # Lowercase, remove punctuation and stopwords
    final = [
        token.lower()
        for token in tokens
        if any(char.isalpha() for char in token)
        and token.lower() not in stop_words
    ]
    return final


preprocessing_rows = []

for i, text in enumerate(raw_tickets):
    sentences = sent_tokenize(text)
    tokens = basic_tokenize(text)

    punctuation_tokens = [
        token for token in tokens
        if is_punctuation_token(token)
    ]

    stopword_tokens = [
        token for token in tokens
        if token.lower() in stop_words
    ]

    unique_tokens = set(token.lower() for token in tokens)
    final_tokens = get_final_tokens(text)

    preprocessing_rows.append({
        "Ticket": i + 1,
        "Sentences": len(sentences),
        "Tokens": len(tokens),
        "Punctuation Tokens": len(punctuation_tokens),
        "Stopwords": len(stopword_tokens),
        "Unique Tokens": len(unique_tokens),
        "Final Tokens": len(final_tokens)
    })

preprocessing_df = pd.DataFrame(preprocessing_rows)

print("\nPreprocessing table:")
print(preprocessing_df.to_string(index=False))

print("\nTOTAL COUNTS:")
print(preprocessing_df.drop(columns="Ticket").sum().to_string())

# Display actual tokens for one selected ticket
selected_index = 0
selected_text = raw_tickets[selected_index]
selected_tokens = basic_tokenize(selected_text)

selected_punctuation = [
    token for token in selected_tokens
    if is_punctuation_token(token)
]

selected_stopwords = [
    token for token in selected_tokens
    if token.lower() in stop_words
]

selected_final = get_final_tokens(selected_text)

print(f"\nDetailed preprocessing for Ticket {selected_index + 1}:")
print("Original text:", selected_text)
print("Actual tokens:", selected_tokens)
print("Punctuation tokens:", selected_punctuation)
print("Stopwords:", selected_stopwords)
print("Final tokens:", selected_final)


# ================================================================
# Q2. REGEX EXTRACTION AND REPLACEMENT
# ================================================================

print("\n" + "=" * 75)
print("Q2: REGEX EXTRACTION AND REPLACEMENT")
print("=" * 75)

# TEST_LINES is supplied by the starter notebook.
if "TEST_LINES" not in globals():
    TEST_LINES = [
        "Please contact support@example.com about version 2.0.1.",
        "Our Wi-Fi connection is state-of-the-art.",
        "The ticket costs 95.5 and the USER needs help.",
        "Visit https://example.com or www.example.org for support.",
        "Call +91 9876543210 or email help@company.in."
    ]
    print("Note: Using fallback TEST_LINES because none was found.")

print("\nRegex extraction patterns:")

patterns = {
    "Words longer than 5 letters":
        r"\b[A-Za-z]{6,}\b",

    "Numbers, decimals and versions":
        r"(?<![\w.])\d+(?:\.\d+)+(?![\w.])|(?<![\w.])\d+(?:\.\d+)?(?![\w.])",

    "Capitalized words":
        r"\b[A-Z][A-Za-z]*\b",

    "Words beginning with a vowel":
        r"\b[AEIOUaeiou][A-Za-z]*\b"
}

for i, line in enumerate(TEST_LINES):
    print(f"\nTEST LINE {i + 1}: {line}")

    for description, pattern in patterns.items():
        matches = re.findall(pattern, line)
        print(f"{description}: {matches}")

# Replacement patterns
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
url_pattern = r"\b(?:https?://|www\.)[^\s<>]+"
phone_pattern = r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,5}\)?[-.\s]?)?\d{5,10}(?!\w)"

def replace_sensitive_data(text):
    text = re.sub(url_pattern, "<URL>", text, flags=re.IGNORECASE)
    text = re.sub(email_pattern, "<EMAIL>", text)
    text = re.sub(phone_pattern, "<PHONE>", text)
    return text

print("\nREGEX REPLACEMENT RESULTS:")

for line in TEST_LINES:
    print("\nOriginal:", line)
    print("Replaced:", replace_sensitive_data(line))


# ================================================================
# Q3. CUSTOM TOKENIZATION USING ONE REGEX PATTERN
# ================================================================

print("\n" + "=" * 75)
print("Q3: CUSTOM TOKENIZATION")
print("=" * 75)

# Priority matters: URLs/emails/phones/versions must be recognized
# before ordinary words and punctuation.

CUSTOM_TOKEN_PATTERN = re.compile(
    r"""
    https?://[^\s<>]+
    |www\.[^\s<>]+
    |[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}
    |\+?\d[\d().\-\s]{7,}\d
    |\b\d+(?:\.\d+)+\b
    |\b\d+\.\d+\b
    |[A-Za-z]+(?:['’][A-Za-z]+)+
    |[A-Za-z]+(?:-[A-Za-z]+)+
    |\d+(?:\.\d+)?
    |[A-Za-z]+
    |[^\w\s]
    """,
    re.VERBOSE | re.IGNORECASE
)

def my_tokenize(text):
    return CUSTOM_TOKEN_PATTERN.findall(text)


# Combine TEST_LINES and the actual dataset for testing
test_texts = [str(line) for line in TEST_LINES] + raw_tickets

print("\nCUSTOM TOKENIZER RESULTS:")

for i, text in enumerate(test_texts):
    print(f"\nText {i + 1}: {text}")
    print("Custom tokens:", my_tokenize(text))

# Compare custom tokenization with word_tokenize()
differences = []

for text in test_texts:
    custom_tokens = my_tokenize(text)
    nltk_tokens = word_tokenize(text)

    if custom_tokens != nltk_tokens:
        differences.append({
            "Text": text,
            "Custom tokenizer": custom_tokens,
            "word_tokenize": nltk_tokens
        })

print("\nTOKENIZER DIFFERENCES:")

for difference in differences[:10]:
    print("\nText:", difference["Text"])
    print("Custom:", difference["Custom tokenizer"])
    print("NLTK:", difference["word_tokenize"])

print("\nAt least three differences to discuss:")
difference_examples = [
    (
        "Contractions",
        "A custom regex can preserve don't as one token, "
        "while word_tokenize may split it into do and n't."
    ),
    (
        "Hyphenated words",
        "A custom regex can preserve state-of-the-art as one token, "
        "while word_tokenize may split it around hyphens."
    ),
    (
        "URLs, emails and versions",
        "A custom regex can preserve a URL, email or 2.0.1 as one "
        "token, while general tokenizers may split some components."
    )
]

for title, explanation in difference_examples:
    print(f"\n{title}: {explanation}")

print(
    "\nThe exact observed differences depend on the actual TEST_LINES "
    "and ticket text."
)


# ================================================================
# Q4. STEMMING AND LEMMATIZATION COMPARISON
# ================================================================

print("\n" + "=" * 75)
print("Q4: STEMMING AND LEMMATIZATION")
print("=" * 75)

porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Choose a suffix that occurs in at least two dataset words.
all_alpha_words = [
    token.lower()
    for text in raw_tickets
    for token in my_tokenize(text)
    if token.isalpha()
]

suffix_candidates = ["ing", "ed", "ly", "tion", "s"]
chosen_suffix = None
suffix_examples = []

for suffix in suffix_candidates:
    examples = sorted({
        word for word in all_alpha_words
        if len(word) > len(suffix) + 2 and word.endswith(suffix)
    })

    if len(examples) >= 2:
        chosen_suffix = suffix
        suffix_examples = examples[:5]
        break

if chosen_suffix is None:
    # Fallback when the dataset has no two words sharing candidates
    chosen_suffix = "ing"
    suffix_examples = sorted({
        word for word in all_alpha_words if word.endswith("ing")
    })[:5]

regex_stemmer = RegexpStemmer(chosen_suffix)

# Map POS tags to WordNet POS tags
def get_wordnet_pos(word):
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except LookupError:
        tag = "N"

    return {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV
    }.get(tag, wordnet.NOUN)


def lemmatize_pos(word):
    pos = get_wordnet_pos(word)
    return lemmatizer.lemmatize(word, pos=pos)


unique_words = sorted(set(all_alpha_words))

stemming_rows = []

for word in unique_words:
    stemming_rows.append({
        "Word": word,
        "Porter": porter.stem(word),
        "Lancaster": lancaster.stem(word),
        "Regexp": regex_stemmer.stem(word),
        "WordNet POS Lemma": lemmatize_pos(word)
    })

stemming_df = pd.DataFrame(stemming_rows)

# Show ONLY words for which at least one output differs
different_words_df = stemming_df[
    stemming_df.iloc[:, 1:].nunique(axis=1) > 1
]

print("Chosen RegexpStemmer suffix:", chosen_suffix)
print("Example dataset words ending in this suffix:", suffix_examples)
print(
    "Justification: The chosen suffix occurs in at least two dataset "
    "words when such a suffix is available. RegexpStemmer removes "
    "that literal suffix and does not perform full linguistic analysis."
)

print("\nWords with differing outputs:")
print(different_words_df.to_string(index=False))

print("""
Interpretation:
Porter applies rule-based stemming.
Lancaster is generally more aggressive.
RegexpStemmer removes the selected suffix literally.
WordNet lemmatization uses vocabulary and POS information to produce
dictionary forms.
""")


# ================================================================
# Q5. CORPUS STATISTICS
# ================================================================

print("\n" + "=" * 75)
print("Q5: CORPUS STATISTICS")
print("=" * 75)

# S1: all tokens, including punctuation
S1 = [
    token.lower()
    for text in raw_tickets
    for token in my_tokenize(text)
]

# S2: tokens excluding punctuation
S2 = [
    token for token in S1
    if any(char.isalnum() for char in token)
]

# S3: final tokens after removing punctuation and stopwords
S3 = [
    token.lower()
    for text in raw_tickets
    for token in get_final_tokens(text)
]

# S4: S3 after Porter stemming
S4 = [porter.stem(token) for token in S3]


def corpus_statistics(tokens):
    total = len(tokens)
    unique = len(set(tokens))
    ttr = unique / total if total else 0

    return {
        "Total tokens": total,
        "Unique tokens": unique,
        "TTR": round(ttr, 4)
    }


stats_df = pd.DataFrame({
    "S1: All tokens": corpus_statistics(S1),
    "S2: Without punctuation": corpus_statistics(S2),
    "S3: Final tokens": corpus_statistics(S3),
    "S4: After Porter stemming": corpus_statistics(S4)
}).T

print("\nCorpus statistics:")
print(stats_df.to_string())

# 1. Top 10 words of S1 and S3
print("\nTOP 10 WORDS IN S1:")
print(pd.DataFrame(
    Counter(S1).most_common(10),
    columns=["Word", "Frequency"]
).to_string(index=False))

print("\nTOP 10 WORDS IN S3:")
print(pd.DataFrame(
    Counter(S3).most_common(10),
    columns=["Word", "Frequency"]
).to_string(index=False))

# 2. Plot log(rank) versus log(frequency)
def plot_zipf(tokens, title):
    frequencies = sorted(Counter(tokens).values(), reverse=True)

    if not frequencies:
        print("No tokens available to plot.")
        return

    ranks = np.arange(1, len(frequencies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(np.log(ranks), np.log(frequencies), marker="o")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Frequency)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_zipf(S1, "Zipf Plot: S1 All Tokens")
plot_zipf(S3, "Zipf Plot: S3 Final Tokens")

# 3. Top five bigrams of S1 and S3
def print_top_bigrams(tokens, label):
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    top_bigrams = Counter(bigrams).most_common(5)

    print(f"\nTOP 5 BIGRAMS IN {label}:")

    if top_bigrams:
        for (first, second), frequency in top_bigrams:
            print(f"{first} {second}: {frequency}")
    else:
        print("Not enough tokens to form bigrams.")


print_top_bigrams(S1, "S1")
print_top_bigrams(S3, "S3")


# ================================================================
# Q6. INTERPRETATION
# ================================================================

print("\n" + "=" * 75)
print("Q6: INTERPRETATION")
print("=" * 75)

ttr_s1 = stats_df.loc["S1: All tokens", "TTR"]
ttr_s4 = stats_df.loc["S4: After Porter stemming", "TTR"]

print("\n(a) What does TTR change from S1 to S4 mean?")
print(f"TTR of S1: {ttr_s1}")
print(f"TTR of S4: {ttr_s4}")

if ttr_s4 < ttr_s1:
    print(
        "TTR decreased. Stemming merged some word forms into shared "
        "stems, reducing the number of unique forms relative to tokens."
    )
elif ttr_s4 > ttr_s1:
    print(
        "TTR increased. Compare the token counts and unique forms; "
        "this can occur with short or unusual datasets."
    )
else:
    print(
        "TTR stayed the same. In this dataset, stemming did not change "
        "the unique-to-total token ratio."
    )

print("\n(b) Example where word_tokenize and custom tokenization differ:")
example_found = False

for text in test_texts:
    custom = my_tokenize(text)
    standard = word_tokenize(text)

    if custom != standard:
        print("Text:", text)
        print("word_tokenize:", standard)
        print("Custom tokenizer:", custom)
        example_found = True
        break

if not example_found:
    print(
        "No difference was found in the tested text. Try a ticket "
        "containing a contraction, URL, email or hyphenated word."
    )

print("\n(c) Stemming versus lemmatization:")
print(
    "Stemming may produce a truncated form that is not a real word. "
    "Lemmatization uses linguistic information to return a dictionary "
    "form, although the result depends on POS and context."
)

# Find an actual dataset example where POS-aware lemmatization
# produces a different result from Porter stemming.
lemma_example = None

for word in unique_words:
    lemma = lemmatize_pos(word)
    stem = porter.stem(word)

    if lemma != stem:
        lemma_example = (word, stem, lemma)
        break

if lemma_example:
    word, stem, lemma = lemma_example
    print(
        f"Dataset example: '{word}' -> Porter: '{stem}', "
        f"POS-aware lemma: '{lemma}'."
    )
else:
    print(
        "No suitable differing word was found in the dataset. "
        "Illustrative example: 'was' can be lemmatized to 'be' "
        "with verb POS, whereas stemming can produce 'wa'."
    )

print("\n(d) When is POS-aware lemmatization better?")
print(
    "It is useful when grammatical role and meaning matter. "
    "For example, the verb 'was' should become 'be', not an "
    "arbitrary truncated stem. POS-aware lemmatization helps "
    "preserve meaningful dictionary forms."
)

print("\n" + "=" * 75)
print("ASSIGNMENT 7 COMPLETE")
print("=" * 75)
