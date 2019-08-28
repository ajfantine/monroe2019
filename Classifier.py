import string, os
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import random

class Classifier:
    def __init__(self):
        self.model = load_model("revised-headline-classifier-v2.model")

        train_x, train_y, self.test_x, self.test_y, self.max_seq_len, self.max_index, \
        self.tokenizer = self.preprocess()

        print('max seq len: ', self.max_seq_len)

    def preprocess(self):
        real_headlines = []
        article_df = pd.read_csv('./all-the-news/articles1.csv')
        for i in range(len(article_df)):
            #if article_df.publication[i] == 'Fox News':
            real_headlines.append(article_df.title[i])
        #arrays are not inherently mutable so use index to changes values
        for i in range(len(real_headlines)):
            #you can lowercase an entire string, it doesn't have to be word by word
            real_headlines[i] = "".join(v for v in real_headlines[i] if v not in string.punctuation).lower()
            real_headlines[i] = real_headlines[i].encode("utf8").decode("ascii", 'ignore')
            real_headlines[i] = real_headlines[i].replace("the new york times", "")
            real_headlines[i] = real_headlines[i].replace("breitbart", "")
            #asdf is the ending token for a headline
            #headlines[i] += " asdf"
        random.seed(13)
        random.shuffle(real_headlines)
        print('num real headlines: ', len(real_headlines))
        #print(real_headlines[:10])


        files = ['1', '2', '3', '4', '5']
        fake_headlines = []
        for f in files:
            filename = open('gen_headlines_v2_' + f + '.txt', 'r')
            raw = ''.join([word for word in filename])
            fh_subset = raw.split('\n')
            fh_subset.pop(len(fh_subset)-1)
            fake_headlines += fh_subset
        random.shuffle(fake_headlines)
        print('number of fake headlines: ', len(fake_headlines))
        #print(fake_headlines[:10])

        headline_dict = OrderedDict()
        for line in real_headlines:
            headline_dict[line] = 1

        #fake_key = OrderedDict()
        for line in fake_headlines:
            headline_dict[line] = 0

        keys = list(headline_dict.keys())
        #print(type(keys))

        random.shuffle(keys)
        random_headline_dict = OrderedDict()
        for k in keys:
            random_headline_dict[k] = headline_dict[k]

        #print(random_headline_dict)

        x = []
        y = []
        for line, label in random_headline_dict.items():
            x.append(line)
            y.append(label)

        #print(x[:10])
        #print(y[:10])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)

        new_x = []
        for line in x:
            token_list = tokenizer.texts_to_sequences([line])[0]
            new_x.append(token_list)

        max_seq_len = max([len(seq) for seq in new_x])
        #print(max_seq_len)
        #print(new_x[0])
        inp_x = np.array(pad_sequences(new_x, maxlen=max_seq_len))

        max_index = max([index for word, index in tokenizer.word_index.items()])

        #print(inp_x[0])
        index = int(.9 * len(x))

        train_x = inp_x[:index]
        test_x = inp_x[index:]

        train_y = np.array(y[:index])
        test_y = np.array(y[index:])

        return train_x, train_y, test_x, test_y, max_seq_len, max_index, tokenizer


    def build_model(self, x, y):
        total_words = len(tokenizer.word_index) + 1

        model = Sequential()

        model.add(Embedding(input_dim = total_words, output_dim = 128, input_length = max_seq_len))
        #model.add(Dense(50, input_dim=max_seq_len, activation='relu'))
        model.add(LSTM(units=128))

        model.add(Dropout(0.2))
        #model.add(Dense(100, input_dim = 50, activation = 'relu'))

        #model.add(Dense(50, input_dim = 100, activation = 'relu'))

        model.add(Dense(1, activation= 'sigmoid'))

        #v1 optimizer was rmsprop
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        #model.summary()

        model.fit(x, y, epochs=50, batch_size=100, verbose = 2)

        model.save('revised-headline-classifier-v2.model')

    def print_sent(self, tokenizer, sentences, max_index):
        for sent in sentences:
            #print(sent)
            complete = ""
            for token in sent:
                #print(token)
                if token != 0:
                    #print("normalized token: ", token)
                    #token *= max_index
                    #token = int(token)
                    #print("denormalized token: ", token)
                    for word, index in tokenizer.word_index.items():
                        if index == token:
                            complete += word + " "
            #print('complete: ', complete)
            return complete

    '''
    predicted: array of predictions for the data
    x: the input component of the data
    y: the labels for the data, 1 if real, 0 if false
    tokenizer: the tokenizer used for matching word to ints, padding sequences, etc
    max_index: the max index of the word:int dict, used for denormalization
    '''
    def generate_error_report(self, predicted, x, y, tokenizer, max_index, false_pos_only):
        false_neg = []
        false_pos = []
        for i in range(len(predicted)):
            full_info = [0,0,0]
            pred = predicted[i]
            if pred < .5:
                pred = 0
            elif pred >= .5:
                pred = 1
            if pred != y[i]:
                complete = ""

                if pred == 0 and y[i] == 1:
                    if false_pos_only:
                        pass
                    else:

                        full_info[0] = predicted[i]

                        full_info[1] = y[i]
                        #print('x[i]: ', x[i])
                        full_info[2] = self.print_sent(tokenizer, [x[i]], max_index)
                        false_neg.append(full_info)

                elif pred == 1 and y[i] == 0:
                    #print('predicted[i]: ', predicted[i])
                    full_info[0] = predicted[i][0]
                    #print('actual: ', y[i])
                    full_info[1] = y[i]
                    #print('x[i]: ', x[i][0])
                    #full_info[2] = self.print_sent(tokenizer, [x[i][0]], max_index)
                    full_info[2] = x[i][0] + ': ' + str(x[i][1])
                    #print('sent: ', self.print_sent(tokenizer, [x[i]], max_index))
                    #print('full info: ', full_info)
                    if full_info not in false_pos:
                        false_pos.append(full_info)

        if false_pos_only:
            #for item in false_pos:
                #print('the classifier said ', item[0], ' but the actual was ', item[1])
                #print(item[2])
                #print("")
            return false_pos
        return false_pos, false_neg

if __name__ == '__main__':
    pass
