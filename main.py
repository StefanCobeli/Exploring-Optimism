import argparse
import configparser
import sys
# sys.path.insert(0, './code/')
# sys.path.insert(0, './config/')

from CustomTokenizer  import *
from data_preparation import *
from Embedder         import *
from models           import *
from training         import *


if __name__ == '__main__':
    #https://github.com/minqi/learning-to-communicate-pytorch/blob/master/main.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, help='path to existing options file')
    args = parser.parse_args()

    #Parse Configuration file
    config_path = args.config_path
    config      = configparser.ConfigParser()
    config.read(config_path)#'./code/config/OPT.ini')

    BATCH_SIZE          = config.getint('Training', 'BATCH_SIZE')
    NUM_EPOCHS          = config.getint('Training', 'NUM_EPOCHS')
    MODEL_NAME          = config.get('Training', 'MODEL_NAME')
    SETTING_1M1         = config.getboolean('Training', 'SETTING_1M1')

    DATA_PATH           = config.get('Paths', 'DATA_PATH')
    EMBEDDINGS_PATH     = config.get('Paths', 'EMBEDDINGS_PATH')
    HISTORY_PATH        = config.get('Paths', 'HISTORY_PATH')
    LOGGING_PATH        = config.get('Paths', 'LOGGING_PATH')
    MODELS_PATH         = config.get('Paths', 'MODELS_PATH')
    SENTIMENT_PATH      = config.get('Paths', 'SENTIMENT_PATH')


    PRE_TRAINING_ON_TSA = config.getboolean('Sentiment', 'PRE_TRAINING_ON_TSA')
    SENTIMENT_LABEL     = config.get('Sentiment', 'SENTIMENT_LABEL')
    SENTIMENT_TEXT      = config.get('Sentiment', 'SENTIMENT_TEXT')

    #Project pipeline:
    if LOGGING_PATH != "":
        sys.stdout = open(LOGGING_PATH, 'w')
    # else:
    #     sys.stdout = sys.__stdout__
    #     pass

    #Read data: list of tweets & list of golden lables:
    opt_tweets, opt_gold_labels = read_OPT_data(DATA_PATH)
    #Define tokenizer:
    custom_tokenizer    = CustomTokenizer()#.fit_on_texts(tweets)

    if PRE_TRAINING_ON_TSA:
        sent_tweets, sent_gold_labels = read_OPT_data(data_path=SENTIMENT_PATH\
                                           , text_column=SENTIMENT_TEXT\
                                           , label_column=SENTIMENT_LABEL)
        custom_tokenizer    = custom_tokenizer.fit_on_texts(opt_tweets\
                                                            + sent_tweets)
        #Tokenize data using the tokenizer:
        sent_vectorized_tweets, sent_gold_labels = vectorize_data(\
                                                       sent_tweets\
                                                     , sent_gold_labels\
                                                     , custom_tokenizer\
                                                     , MAX_SEQUENCE_LENGTH=40)
    else:
        custom_tokenizer    = custom_tokenizer.fit_on_texts(opt_tweets)
        #Tokenize data using the tokenizer:

    opt_vectorized_tweets, opt_gold_labels = vectorize_data(opt_tweets\
                                                 , opt_gold_labels\
                                                 , custom_tokenizer)

    #If we ignore tweets with annotation in (-1, 1):
    if SETTING_1M1:
        opt_vectorized_tweets, opt_gold_labels = remove_vague_tweets(\
                                                                opt_vectorized_tweets\
                                                              , opt_gold_labels)
    #Binarize gold_labels:
    opt_gold_labels = binarize_labels(opt_gold_labels\
                                      , max_negative_value=0)


    if PRE_TRAINING_ON_TSA:
        x_train, y_train = sent_vectorized_tweets, sent_gold_labels
        x_dev, y_dev     = opt_vectorized_tweets, opt_gold_labels
        x_test, y_test   = opt_vectorized_tweets, opt_gold_labels

    else:
        #Train/Dev/Test split:
        x_train, y_train, x_dev, y_dev, x_test, y_test = \
            train_dev_test_split(opt_vectorized_tweets\
                                 , opt_gold_labels)

    #Load pretrained Embeddings:
    embedder = Embedder(EMBEDDINGS_PATH)
    embedding_layer = embedder.build_embedding_layers(custom_tokenizer,\
                                                     trainable=True)

    #model architecture:
    model_builder = get_model_builder(MODEL_NAME)
    model         = model_builder(embedding_layer)


    #Train model:
    training_history, model = train_model(model=model\
                                          , x_train = x_train\
                                          , y_train = y_train\
                                          , x_dev   = x_dev\
                                          , y_dev   = y_dev\
                                          , epochs      = NUM_EPOCHS\
                                          , batch_size  = BATCH_SIZE\
                                          , models_path = MODELS_PATH\
                                          , histories_path=HISTORY_PATH)

    # return training_history, model
