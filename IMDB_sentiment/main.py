import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from argparse import Namespace

from preprocess import reduce_and_split, preprocess_text
from dataset import IMDB_Dataset
from vocabulary import Vocabulary
from RNN import RNN


args = Namespace(
    raw_dataset_csv = "/Users/samhsia/Pytorch_practice/NLP/IMDB_Dataset.csv",
    proportion = 0.1,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    seed = 111,
    final_csv="data/IMDB/reviews_with_splits.csv"
)



def main():
    raw_dataset = pd.read_csv(args.raw_dataset_csv)
    subset_dataset = reduce_and_split(raw_dataset, args.proportion, args.train_proportion, args.val_proportion, args.test_proportion)
    subset_dataset.review = subset_dataset.review.apply(preprocess_text)

    vocab = Vocabulary()
    vocab_in_index_dataset = vocab.vocab_to_index(subset_dataset)
    # print(len(vocab.vocab))
    train_loader, valid_loader, test_loader = IMDB_Dataset(vocab_in_index_dataset, batch_size=50)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab.vocab) # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
    batch_size = 50

    counter = 0
    print_every = 100
    clip=5 # gradient clipping
    # move model to GPU, if available

    # train for some number of epochs
    net = net.to(device)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, labels = inputs.to(device), labels.to(device)

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

if __name__ == "__main__":
    main()
