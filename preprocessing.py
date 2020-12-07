import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import re


def preProcessing():
    dftrain = pd.read_csv('datasets/train.csv')
    # sns.countplot(df.label)
    # plt.xlabel('Label')
    # plt.title('Sarcasm vs Non-sarcasm')
    dftrain['tweets'] = dftrain['tweets'].apply(lambda x: x.lower())
    dftrain['tweets'] = dftrain['tweets'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
    for idx, row in dftrain.iterrows():
        row[0] = row[0].replace('rt', ' ')

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(dftrain['tweets'].values)
    Xtrain = tokenizer.texts_to_sequences(dftrain['tweets'].values)
    Xtrain = pad_sequences(Xtrain)

    Y = pd.get_dummies(dftrain['label']).values
    X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Y, test_size=0.25, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)


    embed_dim = 128
    lstm_out = 196
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=Xtrain.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    SVG(model_to_dot(model).create(prog='dot', format='svg'))

    batch_size = 32
    history = model.fit(X_train, Y_train, epochs=5, batch_size=batch_size, verbose=2)

    validation_size = 1500

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_loss.png')

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_validate)):

        result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_validate[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("Sarcasm_acc", pos_correct/pos_cnt*100, "%")
    print("Non-Sarcasm_acc", neg_correct/neg_cnt*100, "%")


preProcessing()