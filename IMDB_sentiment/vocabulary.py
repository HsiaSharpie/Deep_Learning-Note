import numpy as np
import pandas as pd
import string
from collections import defaultdict, Counter

class Vocabulary(object):
    def __init__(self):
        self.index_to_token = {}
        self.token_to_index = {}
        self.vocab = []

    def summarize_vocab(self, dataset):
        word_counts = Counter()
        for review in dataset.review:
            for word in review.split(' '):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            self.add_token(word)

        return word_counts


    def add_token(self, word):
        len_word = len(self.vocab)
        self.vocab.append(word)
        self.index_to_token[len_word + 1] = word
        self.token_to_index[word] = len_word + 1


    def zero_padding(self, review, seq_length=200):
        new_review_index = np.zeros(seq_length)
        if len(review) >= seq_length:
            new_review_index[:seq_length] = review[:200]
        else:
            new_review_index[:len(review)] = review

        return new_review_index.tolist()

    def sentiment_to_dummy(self, reviews):
        for review in reviews:
            if review[-2] == 'positive':
                review[-2] = 1
            elif review[-2] == 'negative':
                review[-2] = 0

        return reviews


    def vocab_to_index(self, dataset):
        # Count word's occurence
        self.word_counts = self.summarize_vocab(dataset)

        reviews_in_index = []
        for _, row in dataset.iterrows():
            review_index = []
            for word in row.review.split(' '):
                if word not in string.punctuation:
                    try:
                        review_index.append(self.token_to_index[word])
                    except:
                        print('There are no word {} in here!'.format(word))
            review_index.append(row.sentiment)
            review_index.append(row.split)
            reviews_in_index.append(review_index)

        equal_length_reviews = []
        for review in reviews_in_index:
            equal_length_review = self.zero_padding(review[:-2])
            equal_length_review.extend(review[-2:])
            equal_length_reviews.append(equal_length_review)

        reviews = self.sentiment_to_dummy(equal_length_reviews)
        return reviews
