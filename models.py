from tensorflow.keras.layers import Concatenate, Bidirectional, Conv1D, Dense, Dropout
from tensorflow.keras.layers import Flatten, Input, GRU, LSTM, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from text_classification_with_transformer import *

import numpy as np
# from keras.initializers import Constant

def get_model_builder(model_name="GRUstack_model"):
    if model_name == "BiLSTM_model":
        return BiLSTM_model
    if model_name == "CNN_model":
        return CNN_model
    if model_name == "GRUstack_model":
        return GRUstack_model
    if model_name == "Transformer_model":
        return Transformer_model

def Transformer_model(embedding_layer, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    print('Building Transformer model...')
    num_heads = 5  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    # train a Bidirectional LSTM:
    sequence_input = Input(shape=(embedding_layer.input_length, )\
                           , dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embed_dim  = embedded_sequences.shape[-1]
    maxlen     = embedding_layer.get_config()["input_length"]
    vocab_size = embedding_layer.get_config()["input_dim"]

    # print(embedded_sequences.shape[-1])
    # embedded_sequences = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(embedded_sequences)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(embedded_sequences)#x)
    # x = GlobalAveragePooling1D()(x)

    tweet_branch = x
    tweet_branch = Flatten()(tweet_branch)
    # tweet_branch = Bidirectional(LSTM(256))(embedded_sequences)

    dense_branch = Dense(300, activation='relu')(tweet_branch)
    dense_branch = Dropout(rate = .2)(dense_branch)
    dense_branch = Dense(200, activation='relu')(dense_branch)
    dense_branch = Dense(100, activation='relu')(dense_branch)
    dense_branch = Dropout(rate = .1)(dense_branch)
    preds = Dense(2, activation='softmax')(dense_branch)

    model = Model([sequence_input], preds)
    return model

def BiLSTM_model(embedding_layer, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    print('Building BiLSTM model...')

    # train a Bidirectional LSTM:
    sequence_input = Input(shape=(embedding_layer.input_length, )\
                           , dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    tweet_branch = Bidirectional(LSTM(256))(embedded_sequences)

    dense_branch = Dense(300, activation='relu')(tweet_branch)
    dense_branch = Dropout(rate = .2)(dense_branch)
    dense_branch = Dense(200, activation='relu')(dense_branch)
    dense_branch = Dense(100, activation='relu')(dense_branch)
    dense_branch = Dropout(rate = .1)(dense_branch)
    preds = Dense(2, activation='softmax')(dense_branch)

    model = Model([sequence_input], preds)
    return model

def CNN_model(embedding_layer, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    print('Building CNN model...')

    # train a Bidirectional LSTM:
    sequence_input = Input(shape=(embedding_layer.input_length, )\
                           , dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    tweet_branch = Conv1D(256, 3, activation='relu')(embedded_sequences)
    tweet_branch = MaxPooling1D(3)(tweet_branch)
    # tweet_branch = Conv1D(128, 3, activation='relu')(embedded_sequences)
    # tweet_branch = MaxPooling1D(3)(tweet_branch)
    # tweet_branch = Conv1D(128, 5, activation='relu')(tweet_branch)
    # tweet_branch = MaxPooling1D(5)(tweet_branch)
    tweet_branch = Flatten()(tweet_branch)
    dense_branch = Dense(300, activation='relu')(tweet_branch)
    dense_branch = Dropout(rate = .2)(dense_branch)
    dense_branch = Dense(200, activation='relu')(dense_branch)
    dense_branch = Dense(100, activation='relu')(dense_branch)
    dense_branch = Dropout(rate = .1)(dense_branch)
    preds = Dense(2, activation='softmax')(dense_branch)

    model = Model([sequence_input], preds)
    return model


def GRUstack_model(embedding_layer, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    print('Building GRUstack model...')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(embedding_layer.input_length, )\
                           , dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)
#     x = Conv1D(256, 3, activation='relu')(embedded_sequences)
#     x = MaxPooling1D(3)(x)
#     x = Conv1D(128, 5, activation='relu')(x)
#     x = MaxPooling1D(5)(x)

    x = GRU(256, return_sequences=True)(embedded_sequences)
    x = GRU(128)(x)

#     x = Conv1D(64, 3, activation='relu')(x)
#     x = MaxPooling1D(3)(x)

#     x = LSTM(256)(x)

    tweet_branch = x
    dense_branch = tweet_branch


    # print('2. Dense layers.')

#     dense_branch = Concatenate()([hot_branch, phrase_branch])#.shape
#     dense_branch = Flatten()(tweet_branch)

#     dense_branch = Dense(512, activation='relu')(dense_branch)
#     dense_branch = Dropout(rate = .5)(dense_branch)

    dense_branch = Dense(300, activation='relu')(dense_branch)
    dense_branch = Dropout(rate = .2)(dense_branch)

    dense_branch = Dense(200, activation='relu')(dense_branch)
    dense_branch = Dense(100, activation='relu')(dense_branch)
    dense_branch = Dropout(rate = .1)(dense_branch)

    preds = Dense(2, activation='softmax')(dense_branch)

    model = Model([sequence_input], preds)

    return model

# model = build_model(embedding_layer)
