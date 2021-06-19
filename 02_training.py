import csv #training.py
import os
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense , Dropout, Activation, Permute, Multiply,SimpleRNN
from keras.layers.normalization import BatchNormalization
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from attention_decoder import AttentionDecoder
from keras import backend as K

file = "Iban.tsv"
folder = "Corpus"
# load a clean dataset
def load_clean_sample_data(folder, file):
    lines_filepath = os.path.join(folder,file)
    with open(lines_filepath, "r", encoding="utf8") as read:
        reader = csv.reader(read,delimiter="\t")
        dataset = []
        for row in reader:
            dataset.append(row) 
    read.close()     
    return dataset

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # intereply encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1(y_true, y_pred):
    result_precision = precision(y_true, y_pred)
    result_recall = recall(y_true, y_pred)
    return 2*((result_precision*result_recall)/(result_precision+result_recall+K.epsilon()))

# Building Model
def modelSeq2Seq(vocab, timesteps, n_units, encoder, decoder, attention):
    model = Sequential()
    model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
    if(encoder == "LSTM"):
        model.add(LSTM(n_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    
    model.add(RepeatVector(timesteps))
    if(decoder == "LSTM"):
        model.add(LSTM(n_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    
    model.add(BatchNormalization())
    if(attention == "ATTNDECODER"):
        model.add(AttentionDecoder(n_units, vocab))

    return model

#load datasets
dataset = load_clean_sample_data("Corpus","Iban.tsv")
dataset = np.reshape(dataset, (-1,2))
dataset1 = dataset.reshape(-1,1)
train, test = dataset[ : int(len(dataset)*80/100) ],  dataset[ int(len(dataset)*80/100): ]
del dataset

#Tokenizer
all_tokenizer = create_tokenizer(dataset1[:, 0])
all_vocab_size = len(all_tokenizer.word_index) + 1
all_length = max_length(dataset1[:, 0])
print('ALL Vocabulary Size: %d' % (all_vocab_size))
print('ALL Max question length: %d' % (all_length))
del dataset1

# Training Data
trainX = encode_sequences(all_tokenizer, all_length, train[:, 0])
trainY = encode_sequences(all_tokenizer, all_length, train[:, 1])
trainY = encode_output(trainY, all_vocab_size)
del train

# Testing Data
testX = encode_sequences(all_tokenizer, all_length, test[:, 0])
testY = encode_sequences(all_tokenizer, all_length, test[:, 1])
testY = encode_output(testY, all_vocab_size)
del test

# Start Train by Model
encoder = "LSTM"
decoder = "LSTM"
attention = "ATTNDECODER"
model = modelSeq2Seq(all_vocab_size, all_length, 256, encoder, decoder, attention)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy', precision, recall, f1])


# Summarize Defined Model
tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#train and save model
import timeit
start_time = timeit.default_timer()
history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_data=(testX, testY), verbose=1,callbacks=[tensor_board])
print(timeit.default_timer() - start_time)

import timeit
start_time = timeit.default_timer()
score = model.evaluate(testX, testY, batch_size=32)
print(timeit.default_timer() - start_time)
print(score)

import datetime
str(datetime.timedelta(seconds=3437))

filename = 'Model/MODEL_'+encoder+'_'+decoder+'_'+attention+'.h5'
model.save(filename)