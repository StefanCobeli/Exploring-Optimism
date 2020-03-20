from keras.initializers import Constant
from keras.layers       import Embedding
import numpy as np
import os


class Embedder:
    def __init__(self, path_to_emb=None):
        # , emb_size=50,\ ):
        emb_name = path_to_emb.split("/")[-1]
        print('Indexing pre-trained word vectors...')
        print("Using %s:" %emb_name)

        self.path_to_emb = path_to_emb
        # self.embeddings_size  = emb_size

        if not(self.path_to_emb):
            print("Using One-Hot-Encode embedding.")
            print("To be implemented...")
            return

        self.embeddings_index = {}
#         "glove.twitter.27B"
#         'glove.6B.%dd.txt'
        # with open(os.path.join(path_to_emb\
        # , "glove.twitter.27B.%dd.txt" %self.embeddings_size)
        with open(path_to_emb, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                self.embeddings_index[word] = coefs

        self.embeddings_size = self.embeddings_index[word].shape[0]

        print('Found %s pre-trained word vectors.' \
        % (f'{len(self.embeddings_index):,}'))
        print("The words will be embedded in %d-dimensional vectors."\
            %self.embeddings_size)


    def build_embedding_layers(self, custom_tokenizer\
                              , trainable=True):
        '''
            We neeed the word_index,
        '''
        EMBEDDING_DIM = self.embeddings_size
        word_index    = custom_tokenizer.get_word_index()
        num_words     = len(word_index) + 1
        # prepare embedding matrix
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        MAX_SEQUENCE_LENGTH     = custom_tokenizer.MAX_SEQUENCE_LENGTH

        #If No pretrained embeddings were provided:
        if self.path_to_emb == False:
            embedding_layer = Embedding(num_words, EMBEDDING_DIM,
                     input_length=MAX_SEQUENCE_LENGTH, trainable=trainable)
            return embedding_layer

        #If GloVe embeddings were provided:
        for word, i in word_index.items():
            if i >= num_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                if embedding_vector.shape[0] != self.embeddings_size:
                    continue
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=trainable)


        print('Embedding layer prepared.\n')

        return embedding_layer



# embedder = Embedder(False)#'./glove.6B/')
# # embedder = Embedder(path_to_emb='./embeddings/glove.6B/',\
# #                    emb_size=300)
# embedder = Embedder(path_to_emb='./embeddings/glove.twitter.27B/',\
#                    emb_size=200)
# len(custom_tokenizer.get_word_index())
# embedding_layer = build_embedding_layers(embedder, custom_tokenizer,\
#                                         trainable=True)
# embedding_layer.input_length
