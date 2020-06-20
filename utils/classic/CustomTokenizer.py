# import numpy as np
from keras.preprocessing.text     import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class CustomTokenizer:
    def __init__(self, MAX_SEQUENCE_LENGTH=40):
        self.MAX_SEQUENCE_LENGTH     = MAX_SEQUENCE_LENGTH
        self.tokenizer = Tokenizer()
        print("Tokenizer created!")
#         print("TTT %s" %ana)

    def fit_on_texts(self, texts):
#         self.MAX_SEQUENCE_LENGTH = 1 + max(map(lambda x: \
#                                                len(x.split(" "))\
#                                            , texts))
        self.tokenizer.fit_on_texts(texts)
        print("Tokenizer fitted on %s texts.\n" \
            %(f'{len(texts):,}'))
        return self

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def get_word_index(self):
        return self.tokenizer.word_index

    def pad_sequences(self, sequences, maxlen):
        #keras pad_sequence method:
        return pad_sequences(sequences, maxlen=maxlen)

# tweets = ["Hello", "Hello, hello again"]
# custom_tokenizer = CustomTokenizer()#2, 3)
# custom_tokenizer.fit_on_texts(tweets)
# custom_tokenizer.get_word_index()["again"]
