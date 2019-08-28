'''
Generates headlines using the first file in the all-the-news dataset as training
data. Can generate headlines of various diversities (more randomness in word choices)
and has endline tokens built into the training so headlines are of various lengths.

Author: Alex Fantine
'''

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
import keras.utils as ku #used to encode one hot vectors for label data

import random
import pandas as pd #for use with text preprocessing
import numpy as np
import string, os

class Generator:
    def __init__(self):
        #np.random.seed(2) #set the seed for deterministic generation
        self.tokenizer, self.max_seq_len, self.headlines = self.preprocess()

        self.model = load_model("headline-generator-v6.model")

    #preprocess headlines data and splits each headline into a number of sequences
    def preprocess(self):
        headlines = []
        article_df = pd.read_csv('./all-the-news/articles1.csv')
        for i in range(len(article_df)):
            headlines.append(article_df.title[i])

        #arrays are not inherently mutable so use index to changes values
        for i in range(len(headlines)):
            #you can lowercase an entire string, it doesn't have to be word by word
            headlines[i] = "".join(v for v in headlines[i] if v not in string.punctuation).lower()
            headlines[i] = headlines[i].encode("utf8").decode("ascii", 'ignore')
            headlines[i] = headlines[i].replace("the new york times", "")
            headlines[i] = headlines[i].replace("breitbart", "")
            #asdf is the ending token for a headline
            headlines[i] += " asdf"

        random.seed(13)
        random.shuffle(headlines)
        updated = int(len(headlines)/2)

        headlines = headlines[:updated]
        tokenizer = Tokenizer()

        tokenizer.fit_on_texts(headlines) #creates an integer:word dictionary
        total_words = len(tokenizer.word_index) + 1

        input_sequences = []
        for line in headlines:
            #turns each headline into its corresponding integer sequence
            token_list = tokenizer.texts_to_sequences([line])[0]

            #splits each headline into a sequence, so [1,2,3] would become [1,2][1,2,3]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        #data must be padded to be accepted by keras, AKA all sequences must have same length
        max_seq_len = max([len(seq) for seq in input_sequences])

        #pad_sequences(input, maxlen, padding) padding defaults to pre
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len))

        #split into data and labels, where the label is the last word in the sequence
        predictors, label = input_sequences[:,:-1], input_sequences[:,-1]

        #transforms only the label data into one hot vectors
        label = ku.to_categorical(label, num_classes=total_words)

        return tokenizer, max_seq_len, headlines


    def model(self):
        #input len is max seq - 1 because the largest sequences is all of the words
        #in the longest headline, minus the last word, which is the label
        input_len = max_seq_len-1
        total_words = len(tokenizer.word_index) + 1

        model = Sequential()

        #number of distinct words in training set, size of embedding vectors,
        #size of each input sequence
        model.add(Embedding(total_words, 50, input_length = input_len))

        model.add(LSTM(100, return_sequences = True))

        model.add(LSTM(100))

        model.add(Dropout(0.2))

        model.add(Dense(100, activation= 'relu'))

        #softmax squashes the values between 0 and 1 but makes sure they add to 1
        #the values are proprtional to the number of values
        model.add(Dense(total_words, activation='softmax'))

        #calculates with an ouput between 0 and 1
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics =['acc'])

        model.fit(predictors, label, batch_size=128, epochs=50, verbose=1)

        #v4 is pretty much trash, even after testing :(
        #v5 doesn't seem much better... same architecture with normalized vectors
        #v6 was trained with two LSTM layers and two dense layers, 50 epochs and batch size of 128
        #reached a loss of 3.6323 acc of 0.3197, not the worst model! will gen fake headlines with this
        model.save("headline-generator-v6.model")


    def sample(self, preds, diversity, experiments):
          #turns preds into a numpy array continaing 64 bit floats
          #this is to allows for the larger math later on and keep
          #the array values consistent
          preds = np.asarray(preds).astype('float64')

          #the diversity value must be factored into the values
          #np.log takes the natural log of each element in peds
          #np.exp finds the exponential of each element, which is just x^e where e~2.7
          #after some math, these operations can be condensed into one operation
          exp_preds = preds ** (1.0/diversity)

          #normalizes each element based on sum of exp_preds
          preds = exp_preds / np.sum(exp_preds)

          #takes experiments with one of p possible outcomes, in this case, preds is the
          #probabilites of each of the p different outcomes, p is equal to size of vocab
          #the more experiments performed, the more likely the new diverse probabilities
          #get closer to the original probabilities
          #probas is an array of mostly zeroes and a one, which represents the word that
          #the multinomial distribution chose
          probas = np.random.multinomial(experiments, preds, 1)

          #this just returns the index of that one
          return np.argmax(probas)

    '''
    num_words: the length of the generation

    seed_text: the input text

    diversity: the larger this value, the more likely to get different generations

    experiments: the number of experiments to perform in the multinomial distribution,
    the more experiments, the closer to the actual probabilities (more repeats)

    returns: a tuple containing the headline and the number of repeats
    '''
    def generate_headline(self, num_words, seed_text, diversity, experiments=1):
        repeat = 0
        #this for loop iterates the length of the headline you want to generate
        for i in range(num_words):
            #counts the number of repeated phrases in the generation
            for line in self.headlines:
                if seed_text in line:
                    repeat += 1
                    break

            #convert seed text into padded integer sequence
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_seq_len-1)

            predicted = self.model.predict(token_list, verbose=0)[0]

            next_index = self.sample(predicted, diversity[i], experiments)

            for word, index in self.tokenizer.word_index.items():
                if index == next_index:
                    seed_text += " " + word

        seed_list = seed_text.split(' ')
        for i in range(len(seed_list)):
            if seed_list[i] == 'asdf':
                seed_list = seed_list[:i]
                break
        seed_text = ' '.join(seed_list)

        return (seed_text, repeat)

    #mostly used for testing within this class, can be removed and wont affect headline-gan.py
    def set_gen_value(self, num_words, seed_text, diversity, num_headlines, experiments=1):
        generated_headlines = []
        for i in range(num_headlines):
            if len(diversity) == num_words:
                headline = self.generate_headline(num_words, seed_text, diversity, experiments)
                generated_headlines.append(headline)
            else:
                print('Error: number of diversities do not match number of words.')
        return generated_headlines


if __name__ == '__main__':
    pass
