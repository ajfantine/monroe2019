'''headline-gan.py

Author: Alex Fantine

This program is meant to simulate a General Adversarial Network, where the classifier
is used to improve the generator. Mainly, I'm trying to find an ideal diversity value
to use during generation such that the classifier predicts the most false positives.
This is only a simulation of a GAN because neither the classifier or the generator
are being updated after each pass, rather I'm just using two models to find the optimal
value for one variable.

The Classifier object has access to the following variables and methods:
self.model
self.test_x, self.test_y (for testing purposes only, since new data should be provided)
self.tokenizer
self.max_index (for denormalization)

self.generate_error_report(predicted, test_x, test_y, tokenizer, max_index, false_pos_only)
where predicted is the result from the model, and x is the input values and y are the labels
and false_pos_only is set to true if only printing the false positives

returns false_pos, false_neg (only false_pos if false_pos_only is true)
where each element is a list containing [predicted prob, actual value, string headline]
'''

from Classifier import Classifier
from Generator import Generator
from keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
from collections import defaultdict


'''
This method generates 1,000 headlines, where every ten, the input word changes.

generator is the generator object to be used
diversity vals is the array of diversities for each word, will print error if
number of diversities does not match the number of words in the generation.

returns a list of tuples, where each is a headline string and its repeat value
'''
def generate_headlines(generator, diversity_vals, num_headlines, verbose = 0):
    headlines = []
    for i in range(0, num_headlines):
        if verbose == 1:
            print('iteration ' , i+1, '/', num_headlines)
        rando_int = random.randint(0,len(generator.tokenizer.word_index.items()))

        for word, index in generator.tokenizer.word_index.items():
            if index == rando_int:
                rando_word = word
                break
        #num_words, seed_text, diversity, experiments=1
        generated_headline = generator.generate_headline(15, rando_word, diversity_vals)

        headlines.append(generated_headline)
    return headlines

#prints all of the headlines with repeat values
def print_headlines(generated_headlines):
    for line in generated_headlines:
        print(line[0], ': ', line[1])

#returns an array of just headline strings, useful for creating input for the classifier
def clean_headlines(generated_headlines):
    cleaned_headlines = []
    for line in generated_headlines:
        cleaned_headlines.append(line[0])
    return cleaned_headlines

def preprocess_headlines(classifier, x):
    #classifier.tokenizer.fit_on_texts(x)

    new_x = []
    for line in x:
        token_list = classifier.tokenizer.texts_to_sequences([line])[0]
        new_x.append(token_list)

    #max_seq_len = max([len(seq) for seq in new_x])

    inp_x = np.array(pad_sequences(new_x, maxlen=classifier.max_seq_len))

    #normalize the data to be between 0 and 1
    #inp_x = inp_x / classifier.max_index

    return inp_x

def print_prediction_info(values):
    for item in values:
        print('the classifier said ', item[0], ' but the actual was ', item[1])
        print(item[2])
        print("")

def perform_gan_step(generator, classifier):
    div_list = []
    random.seed()
    div_val = random.uniform(.5, .9)
    #div_val = 1.626816675757897
    print("")
    print('diversity value: ', div_val)
    for i in range(0, 16):
        div_list.append(div_val)

    #generate 100 headlines with set diversity
    generated_headlines = generate_headlines(generator, div_list, 100, verbose = 0)
    x = clean_headlines(generated_headlines)
    #for item in x:
        #print(item)
    y = [0 for line in range(len(x))]

    X = preprocess_headlines(classifier, x)
    random.shuffle(X)

    predicted = classifier.model.predict_proba(X, verbose=0)

    false_pos = classifier.generate_error_report(predicted, generated_headlines, \
    y, classifier.tokenizer, classifier.max_index, True)


    #print('cleaned headlines: ', x)
    for line in false_pos:
        print(line[2])

    #only measure false pos becuase model receives just fake headlines
    #print('number of false positives: ', len(false_pos))

    #print('number of total headlines: ', len(X))

    #print('percent wrong: ', (len(false_pos)/len(X))*100)

    #print_prediction_info(false_pos)

    return div_val, len(false_pos)

def test_one_diversity(generator, classifier, num_iter=10):
    output_file = open('diversity_experiment.txt', 'a')
    #num_iter = 10
    print("", end='\n', file=output_file)
    for i in range(0,num_iter):
        print("iteration ", i+1, '/', num_iter)
        diversity, incorrect = perform_gan_step(generator, classifier)

        #print(diversity, ': ', incorrect, end='\n', file=output_file)
        #print("")

def get_best_diversities(num_max):
    output_file = open('diversity_experiment.txt', 'r')
    raw_data = ''.join([line for line in output_file])
    raw_data.strip("")
    data = raw_data.split('\n')
    data = [item for item in data if item is not ""]
    val_dict = defaultdict(list)
    for i in range(len(data)):
        vals = data[i].split(' :  ')
        #print(vals)
        val_dict[vals[1]].append(vals[0])
    keys = list(val_dict.keys())
    int_keys = [int(key) for key in keys]
    int_keys.sort(reverse = True)
    #print(val_dict)
    total_averages = []
    for i in range(0,num_max):
        print(int_keys[i], ': ', val_dict[str(int_keys[i])])
        val_dict[str(int_keys[i])] = [float(val) for val in val_dict[str(int_keys[i])]]
        average =  sum(val_dict[str(int_keys[i])])/len(val_dict[str(int_keys[i])])
        print('average of vals: ', average)
        total_averages.append(average)
    print("")
    print('total average: ', sum(total_averages)/len(total_averages))


if __name__ == '__main__':

    generator = Generator()
    classifier = Classifier()
    #random.seed(5) #setting a seed for control when generating sentences of varying diversities
    test_one_diversity(generator, classifier, 5)
    #for i in range(0,5):
    #    test_one_diversity(generator, classifier)

    #get_best_diversities(20)
