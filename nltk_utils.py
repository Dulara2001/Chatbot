import numpy as np
import nltk
from nltk.stem import PorterStemmer
import jellyfish  # Library for Soundex and Levenshtein distance

# Initialize the Porter Stemmer from NLTK
stemmer = PorterStemmer()

# Tokenize a sentence using NLTK
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Perform stemming on a given word using the Porter Stemmer
def stem(word):
    return stemmer.stem(word.lower())

# Calculate the Soundex distance between two words
def soundex_distance(word1, word2):
    return jellyfish.soundex(word1) == jellyfish.soundex(word2)

# Calculate the Levenshtein distance between two words
def levenshtein_distance(word1, word2):
    return jellyfish.levenshtein_distance(word1, word2)

# Create a bag of words representation for a tokenized sentence
def bag_of_words(tokenized_sentence, words):
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Initialize a bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        # Check for an exact match
        if w in sentence_words:
            bag[idx] = 1
        else:
            # Handle typos using Soundex and edit distance
            for word in sentence_words:
                if soundex_distance(w, word) and levenshtein_distance(w, word) <= 1:
                    bag[idx] = 1
                    break

    return bag
