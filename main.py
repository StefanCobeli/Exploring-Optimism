import argparse
import configparser
from set_project_seed import *


# sys.path.insert(0, './code/')
# sys.path.insert(0, './config/')

from CustomTokenizer  import *
from data_preparation import *
from Embedder         import *
from models           import *
from training         import *

from shutil           import copyfile


if __name__ == '__main__':
    #https://github.com/minqi/learning-to-communicate-pytorch/blob/master/main.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, help='path to existing options file')
    args = parser.parse_args()

    #Parse Configuration file
    config_path = args.config_path
    config      = configparser.ConfigParser()
    config.read(config_path)#'./code/config/OPT.ini')

    RANDOM_SEED         = config.getint("Misc", "RANDOM_SEED")

    BATCH_SIZE          = config.getint('Training', 'BATCH_SIZE')
    NUM_EPOCHS          = config.getint('Training', 'NUM_EPOCHS')
    MODEL_NAME          = config.get('Training', 'MODEL_NAME')
    SETTING_1M1         = config.getboolean('Training', 'SETTING_1M1')
    TRAINABLE_EMBEDDING = config.getboolean('Training', 'TRAINABLE_EMBEDDING')

    DATA_PATH           = config.get('Paths', 'DATA_PATH')
    EMBEDDINGS_PATH     = config.get('Paths', 'EMBEDDINGS_PATH')
    HISTORY_PATH        = config.get('Paths', 'HISTORY_PATH')
    LOGGING_PATH        = config.get('Paths', 'LOGGING_PATH')
    MODELS_PATH         = config.get('Paths', 'MODELS_PATH')
    SENTIMENT_PATH      = config.get('Paths', 'SENTIMENT_PATH')


    PRE_TRAINING_ON_TSA = config.getboolean('Sentiment', 'PRE_TRAINING_ON_TSA')
    SENTIMENT_LABEL     = config.get('Sentiment', 'SENTIMENT_LABEL')
    SENTIMENT_TEXT      = config.get('Sentiment', 'SENTIMENT_TEXT')
    TRIM                = config.getboolean('Sentiment', 'TRIM')

    #Set random seed for the experiment:
    np.random.seed(RANDOM_SEED)
    # random.seed(RANDOM_SEED)
    print("Random seed set to %d." %RANDOM_SEED)

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
    MAX_SEQUENCE_LENGTH = None


    if PRE_TRAINING_ON_TSA:
        sent_tweets, sent_gold_labels = read_OPT_data(data_path=SENTIMENT_PATH\
                                           , text_column=SENTIMENT_TEXT\
                                           , label_column=SENTIMENT_LABEL)
        custom_tokenizer    = custom_tokenizer.fit_on_texts(opt_tweets\
                                                            + sent_tweets)

        if TRIM:
            MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x.split(" ")), opt_tweets))
        else:
            MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x.split(" ")), sent_tweets))
        #Tokenize data using the tokenizer:
        sent_vectorized_tweets, sent_gold_labels = vectorize_data(\
                                                       sent_tweets\
                                                     , sent_gold_labels\
                                                     , custom_tokenizer\
                                                     , MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
        sent_gold_labels = binarize_labels(sent_gold_labels, max_negative_value=0)
    else:
        custom_tokenizer    = custom_tokenizer.fit_on_texts(opt_tweets)
        #Tokenize data using the tokenizer:

    opt_vectorized_tweets, opt_gold_labels = vectorize_data(opt_tweets\
                                                 , opt_gold_labels\
                                                 , custom_tokenizer\
                                                 , MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

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
                                 , opt_gold_labels, R_SEED=RANDOM_SEED)

    #Load pretrained Embeddings:
    embedder = Embedder(EMBEDDINGS_PATH)
    embedding_layer = embedder.build_embedding_layers(custom_tokenizer,\
                                                     trainable=TRAINABLE_EMBEDDING)

    #model architecture:
    model_builder = get_model_builder(MODEL_NAME)
    model         = model_builder(embedding_layer, RANDOM_SEED)


    #Train model:
    training_history, model = train_model(model=model\
                                          , model_name = MODEL_NAME\
                                          , x_train = x_train\
                                          , y_train = y_train\
                                          , x_dev   = x_dev\
                                          , y_dev   = y_dev\
                                          , epochs      = NUM_EPOCHS\
                                          , batch_size  = BATCH_SIZE\
                                          , models_path = MODELS_PATH\
                                          , histories_path=HISTORY_PATH)
    ##########################################
    #Save trained model & Training history:
    if "val_acc" not in training_history.history:
        val_acc    = "valAcc%.3f"%(training_history.history["val_accuracy"][-1])
        # training_history.history["val_acc"] = training_history.history["val_accuracy"]
    else:
        val_acc    = "valAcc%.3f"%(training_history.history["val_acc"][-1])

    emb_name   = "GloVe" if "glove" in EMBEDDINGS_PATH else "FastText"
    time_stamp = strftime("%H%M%S_%Y%m%d", gmtime())
    time_stamp = "_".join((config_path.split("/")[-1], emb_name, val_acc, time_stamp))

    #Save configuration file:
    config_hist = HISTORY_PATH + "config_%s" %time_stamp
    copyfile(config_path, config_hist)
    print("Configurations saved in %s." \
            %(config_hist))

    #Save training history dictionary:
    training_history_fn = HISTORY_PATH + ("history_%s.csv" %time_stamp)
    pd.DataFrame(training_history.history)\
        .to_csv(training_history_fn)
    print("Training history saved in %s." \
            %(training_history_fn))

    #Save trained model:
    # trained_model_fn = MODELS_PATH + "trained_model_%s.h5" %time_stamp
    # model.save(trained_model_fn)
    # print("Trained model saved in %s." \
    #             %(trained_model_fn))
    # return training_history, model
