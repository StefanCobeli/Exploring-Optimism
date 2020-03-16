from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, GRU#, GlobalMaxPooling1D
from keras.layers import Input, Concatenate, Flatten, Dropout#, Embedding
from keras.models import Model
# from keras.initializers import Constant

def get_model_builder(model_name="GRUstack_model"):
    if model_name == "GRUstack_model":
        return GRUstack_model


def GRUstack_model(embedding_layer):
    print('Building DNN model...')

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
