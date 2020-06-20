from keras.utils import to_categorical
import numpy  as np
import pandas as pd
import torch

# data_path   = "../data/optimism-twitter-data/"
# train_file  = "tweets_annotation.csv"
# MAX_SEQUENCE_LENGTH     = 0
# MAX_HOT_SEQUENCE_LENGTH = 0
print("Loaded data_preparation module!")

def binarize_opt_labels(gold_labels, max_negative_value=0):
    '''
        Binarize labels with respect to max_negative_value threshold.
    '''
    return torch.tensor(np.where(gold_labels<=0, 0, 1).astype(int))

def load_opt_data(opt_df_path, setting_1M1=False):
    """
        Load prepared/splitted OPT data, from path.

        Returns:
            (sentences_train, labels_train
            , sentences_test, labels_test
            , sentences_val, labels_val)
    """
    #Local used data store:
    #opt_data_path = "../../data/optimism-twitter-data/processed/"
    setting_name = "set1M1" if setting_1M1 else "set0"
    opt_df_train = pd.read_csv(opt_df_path + f"optimism_{setting_name}_train.csv")
    opt_df_test  = pd.read_csv(opt_df_path + f"optimism_{setting_name}_test.csv")
    opt_df_val   = pd.read_csv(opt_df_path + f"optimism_{setting_name}_validation.csv")

    sentences_train = opt_df_train.Tweet.values
    labels_train    = binarize_opt_labels(opt_df_train.AverageAnnotation.values)

    sentences_test  = opt_df_test.Tweet.values
    labels_test     = binarize_opt_labels(opt_df_test.AverageAnnotation.values)

    sentences_val   = opt_df_val.Tweet.values
    labels_val      = binarize_opt_labels(opt_df_val.AverageAnnotation.values)

    print("Loaded Optimism data.")

    return (sentences_train, labels_train\
        , sentences_test, labels_test\
        , sentences_val, labels_val)

def read_pre_training(emo_path=None, hate_path=None\
                    , opt1M1_path=None, sent_path=None):
    """
        Read pretraing data, if available.
        Returns:
            sentences -> np. array of objects;
            labesl    -> np. array of ints.
    """
    if emo_path:
        emonet_df = pd.read_table(emo_path\
                              , names=["Tweet", "Emotion"])
        emonet_df["ELabel"] = emonet_df.Emotion.astype('category').cat.codes
        sentences = emonet_df.Tweet.values
        labels    = emonet_df.ELabel.values.astype(int)
        print("Pre-Training on EmoNet, data ready.")
    if hate_path:
        hate_df     = pd.read_csv(hate_path\
                      , names=["ID", "Tweet", "Label"])
        hate_df["HLabel"] = hate_df.Label.astype('category').cat.codes
        sentences         = hate_df.Tweet.values
        labels            = hate_df.HLabel.values.astype(int)
        print("Pre-Training on Hate, data ready.")
    if opt1M1_path:
        opt_df_train          = pd.read_csv(opt1M1_path + "optimism_set0_train.csv")
        opt_df_train          = opt_df_train[np.logical_or(opt_df_train.AverageAnnotation<=-1\
                                                     , opt_df_train.AverageAnnotation>=1)]
        sentences = opt_df_train.Tweet.values
        labels    = binarize_opt_labels(opt_df_train.AverageAnnotation.values)
        print("Pre-Training on Optimism 1/-1, data ready.")

    if sent_path:
        sent_df   = pd.read_csv(sent_path\
                                , error_bad_lines=False)
        sentences = sent_df.SentimentText.values
        labels    = sent_df.Sentiment.values.astype(int)
        print("Pre-Training on Sentiment, data ready.")
    return sentences, labels



def train_test_val_split(sentences, labels\
                         , train_size=.8, test_size=.1\
                         , r_seed=16):
    """
        Split sentences and labels into train test val.
        val_size is considered as 1-train_size-test_size.

        Method used only for pre-training data,
        since optimism data is already kept in different files
        (seel `load_opt_data` method).

        Returns:
            (sentences_train, labels_train
            , sentences_test, labels_test
            , sentences_val, labels_val)
    """
    np.random.seed(r_seed)
    indices = np.arange(len(sentences))
    np.random.shuffle(indices)

    #Shuffle sentences & labels:
    sentences = sentences[indices]
    labels    = labels[indices]

    #Split data
    num_training_samples = int(train_size * len(sentences))
    num_test_samples     = int(test_size  * len(sentences))

    sentences_train = sentences[:num_training_samples]
    labels_train    = labels[:num_training_samples]

    sentences_test  = sentences[num_training_samples:num_training_samples+num_test_samples]
    labels_test     = labels[num_training_samples:num_training_samples+num_test_samples]

    sentences_val   = sentences[num_training_samples+num_test_samples:]
    labels_val      = labels[num_training_samples+num_test_samples:]

    print("Splitted pre-training data.")

    print("Train shapes:", sentences_train.shape, labels_train.shape)
    print("Test shapes:", sentences_test.shape, labels_test.shape)
    print("Validation shapes:", sentences_val.shape, labels_val.shape)

    return (sentences_train, labels_train\
            , sentences_test, labels_test\
            , sentences_val, labels_val)
