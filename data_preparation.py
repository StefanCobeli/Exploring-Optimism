from keras.utils import to_categorical
import numpy  as np
import pandas as pd

# data_path   = "../data/optimism-twitter-data/"
# train_file  = "tweets_annotation.csv"
# MAX_SEQUENCE_LENGTH     = 0
# MAX_HOT_SEQUENCE_LENGTH = 0

def read_OPT_data(data_path\
                , train=True\
                , text_column='Tweet'\
                , label_column="AverageAnnotation"):
    """
        Read data format given at the Word Complexity Estimation
        task, Deep Learning course, FMI UniBuc 2020.

        input:  folder containing given data file;
        output: texts, hot_words, gold_labels
    """
    original_df = pd.read_csv(data_path\
                    , error_bad_lines=False)# + train_file)

    print('Processing text dataset:')

    texts         = list(original_df[text_column]) # list of text samples
    print('Found %s texts.\n' % (f'{len(texts):,}'))

    if train:
        gold_labels   = list(original_df[label_column]) #list of gold labels
        return texts, gold_labels#, original_df

    return texts

# tweets, gold_labels = read_OPT_data(data_path)#[2][:5]


def vectorize_data(texts, gold_labels\
                   , custom_tokenizer\
                  , MAX_SEQUENCE_LENGTH=None):
    """
        Given a list of texts, a list of subsequences and
        a list of labels for each of the texts, vectorie them.

        input:  texts, hot_words, gold_labels
        output: data_sequences, data_hot_sequences, gold_labels
    """

    print("Vectorizing given data...")
    if not(MAX_SEQUENCE_LENGTH):
        MAX_SEQUENCE_LENGTH = max(map(lambda x: len(x.split(" ")), texts)) + 1
    custom_tokenizer.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    print("The longest tweet/text has %d words." \
          %(MAX_SEQUENCE_LENGTH - 1))

    # finally, vectorize the text samples into a 2D integer tensor

    sequences      = custom_tokenizer.texts_to_sequences(texts)
    data_sequences = custom_tokenizer.pad_sequences(sequences\
                                       , maxlen=MAX_SEQUENCE_LENGTH)

    gold_labels = (np.asarray(gold_labels))
    print('Shape of data_sequences tensor:', data_sequences.shape)
    print('Shape of label tensor: %s.\n' %str(gold_labels.shape))

    return data_sequences, gold_labels

# vectorize_data(*read_OPT_data(data_path))[-1][:6]
# data_sequences, gold_labels = vectorize_data(tweets, gold_labels\
#                    , custom_tokenizer)


def remove_vague_tweets(tweets, gold_labels):
    '''
        Remove tweets with labels in (-1, 1):
    '''
    clear_indexes = np.logical_or(gold_labels <= -1\
                                  , gold_labels >= 1)
    print("Removed tweets with AverageAnnotation in (-1, 1).")
    return tweets[clear_indexes], gold_labels[clear_indexes]

def binarize_labels(gold_labels, max_negative_value=0):
    '''
        Binarize and One-Hot Encode labels.
    '''
    gold_labels_binary = np.where(gold_labels<=max_negative_value\
                                    , 0, 1)
    gold_labels_binary = to_categorical(gold_labels_binary)
    print("Binarized labels!")
    return gold_labels_binary


def train_dev_test_split(vectorized_tweets, gold_labels\
                         , DEV_SPLIT  = .1\
                         , TEST_SPLIT = .1\
                         , R_SEED     = 7):

    #Shuffle indicies:
    data_dim = vectorized_tweets.shape[0]
    indices  = np.arange(data_dim)
    np.random.shuffle(indices)

    #Shuffle data accordingly:
    vectorized_tweets = vectorized_tweets[indices]
    gold_labels       = gold_labels[indices]

    #Compute number of Dev & Test samples:
    num_dev_samples  = int(DEV_SPLIT  * data_dim)
    num_test_samples = int(TEST_SPLIT * data_dim)

    #Split data according to the above proportions:
    x_train = vectorized_tweets[:-(num_dev_samples+num_test_samples)]
    y_train = gold_labels[:-(num_dev_samples+num_test_samples)]

    x_dev   = vectorized_tweets[-(num_dev_samples+num_test_samples)\
                                :-num_test_samples]
    y_dev   = gold_labels[-(num_dev_samples+num_test_samples)\
                                :-num_test_samples]

    x_test  = vectorized_tweets[-num_test_samples:]
    y_test  = gold_labels[-num_test_samples:]

    print("Splitted data into Train: %d; Dev: %d; Test: %d.\n"\
            %(data_dim-num_dev_samples-num_test_samples\
            , num_dev_samples, num_test_samples))

    return x_train, y_train, x_dev, y_dev, x_test, y_test
