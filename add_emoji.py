# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:01:05 2019

@author: rahul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import emoji

dataset=pd.read_csv("emojify_data.csv",header=None)
test=pd.read_csv("test_emoji.csv",header=None)
train=pd.read_csv("train_emoji.csv",header=None)

# =============================================================================
# sequences=[]
# labels=[]
# =============================================================================
sentences=train.iloc[:,0]
test_sentences=test.iloc[:,0]
labels=train.iloc[:,1]
test_labels=test.iloc[:,1]

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# =============================================================================
# =============================================================================

tokenizer=Tokenizer(num_words=100,oov_token="<oov>")
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
print(word_index)
sequence=tokenizer.texts_to_sequences(sentences)
print(max([len(x) for x in sequence]))
padded=pad_sequences(sequence,maxlen=10)
test_sequence=tokenizer.texts_to_sequences(test_sentences)
print(max([len(x) for x in test_sequence]))
test_padded=pad_sequences(test_sequence,maxlen=10)

vocab_size = 1000
embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=10),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 50
history = model.fit(padded, labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=1)

example=['i am sad',
         'i am happy',
         'i will do this',
         'not feeling happy',
         'let us play ball',
         'i want apple',
         'i love apple'
         ]
es=tokenizer.texts_to_sequences(example)
ep=pad_sequences(es,maxlen=10)
predicted=model.predict_classes(ep)

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def get_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

for i in range(0,len(example)):
    print(' prediction({}): '.format(predicted[i]) + example[i] + get_emoji(int(predicted[i])))