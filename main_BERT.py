#main method, only logic assembly:
import argparse
import configparser
import os
import torch
import sys
sys.path.insert(0, '../utils/BERT/')


from utils.BERT.data_preparation import *
from utils.BERT.model_logic      import *
from utils.BERT.tokenization     import *

from torch.optim      import Adam
from tqdm             import tqdm_notebook as tqdm


if __name__ == '__main__':
    #https://github.com/minqi/learning-to-communicate-pytorch/blob/master/main.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, help='path to existing options file')
    args = parser.parse_args()

    #Parse Configuration file
    config_path = args.config_path

    # config_path = "../config/BERT/RoBERTa/RoBERTa_Hate_set0"
    config      = configparser.ConfigParser()
    config.read(config_path)

    FREEZE         = config.getboolean("Misc", "FREEZE")
    # SAVE_LOGITS    = config.getboolean("Misc", "SAVE_LOGITS")
    # Considered random seeds:
    RANDOM_SEED    = eval(config.get('Misc', 'RANDOM_SEED'))#[1, 16, 601, 11, 20]

    #../data/EmoNet/Emonet.tsv
    EMONET_PATH    = config.get('Paths', 'EMONET_PATH')
    #../data/Hate/hatespeech-twitter.csv
    HATE_PATH      = config.get('Paths', 'HATE_PATH')
    #../data/optimism-twitter-data/processed/optimism_set0_train.csv
    OPT1M1_PATH    = config.get('Paths', 'OPT1M1_PATH')
    OPT_PATH       = config.get('Paths', 'OPT_PATH')
    #../data/Sentiment-Analysis-Dataset/Sentiment Analysis Dataset.csv
    SENT_PATH      = config.get('Paths', 'SENT_PATH')
    HISTORIES_PATH = config.get('Paths', 'HISTORIES_PATH')
    DATA_STORE     = config.get('Paths', 'DATA_STORE')
    LOGGING_PATH   = config.get('Paths', 'LOGGING_PATH')

    BATCH_SIZE     = config.getint('Training', 'BATCH_SIZE')
    BATCH_SIZE_PT  = config.getint('Training', 'BATCH_SIZE_PT')
    PRE_TRAINING   = config.getboolean('Training', 'PRE_TRAINING')
    SETTING_1M1    = config.getboolean('Training', 'SETTING_1M1')
    MODEL_NAME     = config.get('Training', 'MODEL_NAME')#roberta-base
    NUM_EPOCHS     = config.getint('Training', 'NUM_EPOCHS')

    #Project pipeline:
    if LOGGING_PATH:
        print(f"Strating to log output in {LOGGING_PATH}.")
        sys.stdout = open(LOGGING_PATH, 'w')

    GPU_AVAILABLE = torch.cuda.is_available()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    PRE_TRAINING_NAME = "EmoNet"  if EMONET_PATH \
                    else "Hate"   if HATE_PATH   \
                    else "Opt1M1" if OPT1M1_PATH \
                    else "Sent"   if SENT_PATH   \
                    else None

    print(PRE_TRAINING_NAME)
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('PyTorch:    No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    tokenizer = pick_tokenizer(model_name=MODEL_NAME)

    #################################################################
    #############PRE-Training data-loading procedure:################
    #################################################################
    if PRE_TRAINING:
        #Read Pre-training data:
        sentences_pre, labels_pre = read_pre_training(emo_path=EMONET_PATH   \
                                                  , hate_path=HATE_PATH  \
                                                  , opt1M1_path=OPT1M1_PATH \
                                                  , sent_path=SENT_PATH)
        #Train/test/val split:
        (sentences_pre_train, labels_pre_train\
        , sentences_pre_test, labels_pre_test\
        , sentences_pre_val, labels_pre_val) = train_test_val_split(sentences_pre, labels_pre)
        #Check if inputs they were generated before
        data_store = DATA_STORE
        if os.path.isfile(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_tokenizer"):
            print(f"Loading {PRE_TRAINING_NAME} tokenizations from {data_store}...")
            tokenizer                 = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_tokenizer")
            input_ids_pre_train       = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_input_ids_train")
            attention_masks_pre_train = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_attention_masks_train")
            input_ids_pre_test        = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_input_ids_test")
            attention_masks_pre_test  = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_attention_masks_test")
            input_ids_pre_val         = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_input_ids_val")
            attention_masks_pre_val   = torch.load(data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_attention_masks_val")
    #         print(input_ids_pre_train.shape, attention_masks_pre_train.shape, labels_pre_train.shape)
    #         print(input_ids_pre_test.shape, attention_masks_pre_test.shape, labels_pre_test.shape)
    #         print(input_ids_pre_val.shape, attention_masks_pre_val.shape, labels_pre_val.shape)
        else:
            #Fit the tokenizer and save the obtained tokenizations:
            print(f"Tokenizing pre-training {PRE_TRAINING_NAME} data.")
            #Train encodings:
            input_ids_pre_train, attention_masks_pre_train   = retrieve_data_encodings(sentences=sentences_pre_train\
                                                                           , tokenizer=tokenizer)
            #Test encodings:
            input_ids_pre_test, attention_masks_pre_test = retrieve_data_encodings(sentences=sentences_pre_test\
                                                                               , tokenizer=tokenizer)
            #Validation encodings:
            input_ids_pre_val, attention_masks_pre_val   = retrieve_data_encodings(sentences=sentences_pre_val\
                                                                               , tokenizer=tokenizer)

            #Save tokenizations
            print(f"Saving {PRE_TRAINING_NAME} tokenizations at {data_store}...")
            torch.save(tokenizer, data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_tokenizer")
            torch.save(input_ids_pre_train      , data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_input_ids_train")
            torch.save(attention_masks_pre_train, data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_attention_masks_train")
            torch.save(input_ids_pre_test       , data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_input_ids_test")
            torch.save(attention_masks_pre_test , data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_attention_masks_test")
            torch.save(input_ids_pre_val        , data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_input_ids_val")
            torch.save(attention_masks_pre_val  , data_store + f"{MODEL_NAME}_{PRE_TRAINING_NAME}_attention_masks_val")



    ##########################
    #############3
    ##########################
    #Same data loading procedure when there is no pretraining:
    (sentences_train, labels_train\
            , sentences_test, labels_test\
            , sentences_val, labels_val) = load_opt_data(opt_df_path=OPT_PATH\
                                                         , setting_1M1=SETTING_1M1)



    print("\nTokenizing Optimism data:")
    input_ids_train, attention_masks_train = retrieve_data_encodings(sentences=sentences_train\
                                                                       , tokenizer=tokenizer)
    input_ids_test, attention_masks_test = retrieve_data_encodings(sentences=sentences_test\
                                                                       , tokenizer=tokenizer)
    input_ids_val, attention_masks_val = retrieve_data_encodings(sentences=sentences_val\
                                                                       , tokenizer=tokenizer)
    print("Generated Optimism data inputs/encodings.")


    print(f"Preparing datasets in batches of size {BATCH_SIZE}.")

    if PRE_TRAINING:
        print("Pre-training data inputs/encodings generated.")
        dataloader_pre_train, dataloader_pre_test, dataloader_pre_validation = retrieve_dataloaders(\
                       input_ids_pre_train, attention_masks_pre_train, labels_pre_train \
                     , input_ids_pre_test, attention_masks_pre_test, labels_pre_test  \
                     , input_ids_pre_val, attention_masks_pre_val, labels_pre_val     \
                     , batch_size=BATCH_SIZE_PT)
    dataloader_train, dataloader_test, dataloader_validation = retrieve_dataloaders(\
                       input_ids_train, attention_masks_train, labels_train \
                     , input_ids_test, attention_masks_test, labels_test  \
                     , input_ids_val, attention_masks_val, labels_val     \
                     , batch_size=BATCH_SIZE)

    print("\nDataLoaders prepared!")

    print("Loading model...")
    if PRE_TRAINING:
        num_labels = np.unique(labels_pre_train).shape[0]
    else:
        num_labels = np.unique(labels_train).shape[0]


    model = pick_model(model_name=MODEL_NAME\
                       , num_labels=num_labels)


    #define optimizer:
    optimizer = Adam(model.parameters()\
                    , lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    )


    print(f"Model: {MODEL_NAME} has {model.num_parameters():,} parameters;")
    # for p in params[-2:]:
    if FREEZE:
        for p in list(model.parameters())[:-4]:
            p.requires_grad = False
        print(f"Freezed model: {MODEL_NAME}\'s hidden layers weights;")
    else:
        for p in model.parameters():
            p.requires_grad = True
        print(f"Model: {MODEL_NAME} ready for fine-tunning.")


    # train_model(model, optimizer\
    #             , batch_size=BATCH_SIZE, dataloader_train=dataloader_train     \
    #             , dataloader_test=dataloader_test, dataloader_val=dataloader_validation  \
    #             , num_epochs=1, random_seed=16\
    #             , model_name=MODEL_NAME, weight_decay=False)



    if PRE_TRAINING:
        model_save_path = f"../models/{MODEL_NAME}_pre-trained_{PRE_TRAINING_NAME}"
        
        if os.path.isfile(model_save_path):
            #Change here the final layer depending on the model:
            model = change_model_top_layer(model, MODEL_NAME)
            print(f"Found already trained {MODEL_NAME} on {PRE_TRAINING_NAME}.")
            model.load_state_dict(torch.load(model_save_path))
            if torch.cuda.is_available():
                model.cuda()
            print("Pre-trained weights loaded!")
        else:
            print("\n##############################")
            print(f"### Pre-training on {PRE_TRAINING_NAME}:")
            print("##############################")
            print("\nStarting pre-training procedure on %s..." %PRE_TRAINING_NAME)
            model, df_stats_pre = train_model(model, optimizer\
                            , batch_size=BATCH_SIZE_PT\
                            , dataloader_train=dataloader_pre_train     \
                            , dataloader_test=dataloader_pre_test\
                              , dataloader_val=dataloader_pre_validation  \
                            , num_epochs=1\
                              , random_seed=16\
                        , model_name=MODEL_NAME\
                          , pre_training_name=PRE_TRAINING_NAME\
                        , pre_training=True  \
                        , histories_path=HISTORIES_PATH \
                        , weight_decay=False)
            print("\nPre-training on %s procedure finished!" %PRE_TRAINING_NAME)
            #Change here the final layer depending on the model:
            model = change_model_top_layer(model, MODEL_NAME)
            if torch.cuda.is_available():
                model.cuda()
            torch.save(model.state_dict(), model_save_path)

    else:
        print("\nNo pretraining was performed!")
        model_save_path = f"../models/{MODEL_NAME}_untrained"
        torch.save(model.state_dict(), f"../models/{MODEL_NAME}_untrained")

    mean_test_accuracies = 0
    mean_val_accuracies  = 0
    
    for i, rs in enumerate(RANDOM_SEED):
        iteration_name=f"{i+1}of{len(RANDOM_SEED)}"
        #reload model to initial values:
        print("#######################################")
        print(f"### Run {iteration_name}, using random seed {rs}:###")
        print("#######################################")

        setting_name = "set1M1" if SETTING_1M1 else "set0"
        model.load_state_dict(torch.load(model_save_path))

        model, df_stats = train_model(model=model\
                    , optimizer=optimizer\
                    , batch_size=BATCH_SIZE\
                    , dataloader_train=dataloader_train     \
                    , dataloader_test=dataloader_test\
                      , dataloader_val=dataloader_validation  \
                    , num_epochs=NUM_EPOCHS\
                    , random_seed=rs\
                    , model_name=MODEL_NAME\
                      , pre_training_name=PRE_TRAINING_NAME\
                    , pre_training=False  \
                    , histories_path=HISTORIES_PATH \
                    , iteration=iteration_name \
                    , weight_decay=False
                        , setting=setting_name)
        
        mean_val_accuracies  += (1/len(RANDOM_SEED)) * df_stats['Valid. Accur.'].values[-1]
        mean_test_accuracies += (1/len(RANDOM_SEED)) * df_stats['Test Accur.'].values[-1]

        opt_df_train = pd.read_csv(OPT_PATH + f"optimism_{setting_name}_train.csv")
        opt_df_test  = pd.read_csv(OPT_PATH + f"optimism_{setting_name}_test.csv")
        opt_df_val   = pd.read_csv(OPT_PATH + f"optimism_{setting_name}_validation.csv")

        print("Saving logits:")
        logits_df_train   = retrieve_logits(model        = model            \
                        , opt_df     = opt_df_train     \
                        , dataloader = dataloader_train \
                        , input_ids  = input_ids_train  \
                        , sentences  = sentences_train  \
                        , labels     = labels_train     \
                        , model_name = MODEL_NAME\
                        , batch_size = BATCH_SIZE\
                        , opt_data_path=OPT_PATH\
                        , pre_training_name=PRE_TRAINING_NAME \
                        , iteration=iteration_name
                       , data_type="train"
                        , setting=setting_name)

        logits_df_val   = retrieve_logits(model        = model            \
                , opt_df     = opt_df_val     \
                , dataloader = dataloader_validation \
                , input_ids  = input_ids_val  \
                , sentences  = sentences_val  \
                , labels     = labels_val     \
                , model_name = MODEL_NAME\
                , batch_size = BATCH_SIZE\
                , opt_data_path=OPT_PATH\
                , pre_training_name=PRE_TRAINING_NAME\
                , iteration=iteration_name
                     , data_type="val"
                        , setting=setting_name)
        
        logits_df_test   = retrieve_logits(model        = model            \
                    , opt_df     = opt_df_test     \
                    , dataloader = dataloader_test \
                    , input_ids  = input_ids_test  \
                    , sentences  = sentences_test  \
                    , labels     = labels_test     \
                    , model_name = MODEL_NAME\
                    , batch_size = BATCH_SIZE\
                    , opt_data_path=OPT_PATH\
                    , pre_training_name=PRE_TRAINING_NAME\
                    , iteration=iteration_name
                      , data_type="test"
                        , setting=setting_name)
    print("############################################################")
    print("###################FINAL STATS:#############################")
    print("############################################################")
    print(f"Obtained mean accuracies over the {len(RANDOM_SEED)} runs:")
    print("Mean Validation Accuracy:", mean_val_accuracies)
    print("Mean Test Accuracy:", mean_test_accuracies)