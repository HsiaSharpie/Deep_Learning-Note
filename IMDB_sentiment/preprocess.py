import numpy as np
import pandas as pd
import re
from collections import defaultdict

def reduce_and_split(dataset, proportion, train_proportion, val_proportion, test_proportion):
    # Reduce the dataset to minimal-size
    by_sentiment = defaultdict(list)

    for _, row in dataset.iterrows():
        by_sentiment[row.sentiment].append(row.to_dict())

    reviews_subset = []
    for _, item_list in by_sentiment.items():
        np.random.shuffle(item_list)
        len_reviews = len(item_list)
        len_sub_reviews = int(len_reviews * proportion)

        reviews_subset.extend(item_list[:len_sub_reviews])
    subset_dataset = pd.DataFrame(reviews_subset)


    # split the dataset to different set(train, val, test)
    by_sentiment = defaultdict(list)

    for _, row in subset_dataset.iterrows():
        by_sentiment[row.sentiment].append(row.to_dict())

    subset_list = []
    for _, item_list in by_sentiment.items():
        np.random.shuffle(item_list)

        n_total = len(item_list)
        n_train = int(train_proportion * n_total)
        n_val = int(val_proportion * n_total)
        n_test = int(test_proportion * n_total)

        for item in item_list[:n_train]:
            # item is a dictionary with review, sentiment (keys)
            item['split'] = 'train'

        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'

        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
        subset_list.extend(item_list)
    subset_df = pd.DataFrame(subset_list)

    return subset_df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
