#main method, only logic assembly:
import torch
import sys

from tqdm import tqdm

# import argparse
# import configparser

sys.path.insert(0, '../utils/BERT/')
# sys.path.insert(0, './code/')
# sys.path.insert(0, './config/')

# from set_project_seed import *
from data_preparation import *
from model_logic      import *
from tokenization     import *
import os

from torch.optim import Adam

BATCH_SIZE = 32

GPU_AVAILABLE = torch.cuda.is_available()
EMONET_PATH = None
HATE_PATH   = None
OPT_PATH    = "../../data/optimism-twitter-data/processed/"#/optimism_set0_train.csv"
SENT_PATH   = None

PRE_TRAINING_NAME = "EmoNet"  if EMONET_PATH \
                else "Hate"   if HATE_PATH   \
                else "Opt1M1" if OPT_PATH    \
                else "Sent"   if SENT_PATH   \
                else None
PRE_TRAINING = True
SETTING_1M1  = True
# MODEL_NAME = ['albert-base-v2'\
#               , 'bert-base-uncased', 'bert-large-uncased'\
#               , 'roberta-base', 'xlnet-base-cased',  ]
MODEL_NAME = 'albert-base-v2'

# Best for each model; and 1 for each pre-training dataset.
NUM_EPOCHS = 1
# FREEZE = True
FREEZE = False
HISTORIES_PATH = "."
SAVE_LOGITS    = True

# Considered random seeds:
RANDOM_SEED = [1, 16, 601, 11, 20]
# RANDOM_SEED = 1


DATA_STORE = "../../data/pck_objects/"
#Read dataframes maybe later, when needing logits:
# opt_df_train          = pd.read_csv("../../data/optimism-twitter-data/processed/optimism_set0_train.csv")
# opt_df_train          = pd.read_csv("../../data/optimism-twitter-data/processed/optimism_set1M1_train.csv")

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('PyTorch:    No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#################################################################
#############PRE-Training data-loading procedure:################
#################################################################
if PRE_TRAINING:
    #Read Pre-training data:
    sentences_pre, labels_pre = read_pre_training(emo_path=EMONET_PATH   \
                                              , hate_path=HATE_PATH  \
                                              , opt1M1_path=OPT_PATH \
                                              , sent_path=SENT_PATH)
    #Train/test/val split:
    (sentences_pre_train, labels_pre_train\
    , sentences_pre_test, labels_pre_test\
    , sentences_pre_val, labels_pre_val) = train_test_val_split(sentences_pre, labels_pre)
    #Check if inputs they were generated before
    data_store = DATA_STORE
    if os.path.isfile(data_store + f"{PRE_TRAINING_NAME}_tokenizer"):
        print(f"Loading {PRE_TRAINING_NAME} tokenizations from {data_store}...")
        tokenizer                 = torch.load(data_store + f"{PRE_TRAINING_NAME}_tokenizer")
        input_ids_pre_train       = torch.load(data_store + f"{PRE_TRAINING_NAME}_input_ids_train")
        attention_masks_pre_train = torch.load(data_store + f"{PRE_TRAINING_NAME}_attention_masks_train")
        input_ids_pre_test        = torch.load(data_store + f"{PRE_TRAINING_NAME}_input_ids_test")
        attention_masks_pre_test  = torch.load(data_store + f"{PRE_TRAINING_NAME}_attention_masks_test")
        input_ids_pre_val         = torch.load(data_store + f"{PRE_TRAINING_NAME}_input_ids_val")
        attention_masks_pre_val   = torch.load(data_store + f"{PRE_TRAINING_NAME}_attention_masks_val")
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
        torch.save(tokenizer, data_store + f"{PRE_TRAINING_NAME}_tokenizer")
        torch.save(input_ids_pre_train      , data_store + f"{PRE_TRAINING_NAME}_input_ids_train")
        torch.save(attention_masks_pre_train, data_store + f"{PRE_TRAINING_NAME}_attention_masks_train")
        torch.save(input_ids_pre_test       , data_store + f"{PRE_TRAINING_NAME}_input_ids_test")
        torch.save(attention_masks_pre_test , data_store + f"{PRE_TRAINING_NAME}_attention_masks_test")
        torch.save(input_ids_pre_val        , data_store + f"{PRE_TRAINING_NAME}_input_ids_val")
        torch.save(attention_masks_pre_val  , data_store + f"{PRE_TRAINING_NAME}_attention_masks_val")

    print("Pre-training data inputs/encodings generated.")
    dataloader_pre_train, dataloader_pre_test, dataloader_pre_validation = retrieve_dataloaders(\
                   input_ids_pre_train, attention_masks_pre_train, labels_pre_train \
                 , input_ids_pre_test, attention_masks_pre_test, labels_pre_test  \
                 , input_ids_pre_val, attention_masks_pre_val, labels_pre_val     \
                 , batch_size=128)

##########################
#############3
##########################
#Same data loading procedure when there is no pretraining:
(sentences_train, labels_train\
        , sentences_test, labels_test\
        , sentences_val, labels_val) = load_opt_data(opt_df_path=OPT_PATH\
                                                     , setting_1M1=SETTING_1M1)
tokenizer = pick_tokenizer(model_name=MODEL_NAME)


print("\nTokenizing Optimism data:")
input_ids_train, attention_masks_train = retrieve_data_encodings(sentences=sentences_train\
                                                                   , tokenizer=tokenizer)
input_ids_test, attention_masks_test = retrieve_data_encodings(sentences=sentences_test\
                                                                   , tokenizer=tokenizer)
input_ids_val, attention_masks_val = retrieve_data_encodings(sentences=sentences_val\
                                                                   , tokenizer=tokenizer)
print("Generated Optimism data inputs/encodings.")

print("\nTokenizing ")
input_ids_train, attention_masks_train = retrieve_data_encodings(sentences=sentences_train\
                                                                   , tokenizer=tokenizer)
input_ids_test, attention_masks_test = retrieve_data_encodings(sentences=sentences_test\
                                                                   , tokenizer=tokenizer)
input_ids_val, attention_masks_val = retrieve_data_encodings(sentences=sentences_val\
                                                                   , tokenizer=tokenizer)
print("Generated Optimism data inputs/encodings.")

print(f"Preparing datasets in batches of size {BATCH_SIZE}.")


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
    print(f"\nFreezed model: {MODEL_NAME}\'s hidden layers weights;")
else:
    for p in model.parameters():
        p.requires_grad = True
    print(f"\nModel: {MODEL_NAME} ready for fine-tunning.")


# train_model(model, optimizer\
#             , batch_size=BATCH_SIZE, dataloader_train=dataloader_train     \
#             , dataloader_test=dataloader_test, dataloader_val=dataloader_validation  \
#             , num_epochs=1, random_seed=16\
#             , model_name=MODEL_NAME, weight_decay=False)



if PRE_TRAINING:
    print("\nStarting pre-training procedure on %s..." %PRE_TRAINING_NAME)
    model, df_stats_pre = train_model(model, optimizer\
                    , batch_size=4\
                      #batch_size Should be 128
                    , dataloader_train=dataloader_pre_train     \
                    , dataloader_test=dataloader_pre_test\
                      , dataloader_val=dataloader_pre_validation  \
                    , num_epochs=1\
                      , random_seed=16\
                , model_name=MODEL_NAME\
                  , pre_training_name=PRE_TRAINING_NAME\
                , pre_training=True  \
                , histories_path="." \
                , weight_decay=False)

    model_save_path = f"../../models/{MODEL_NAME}_pre-trained_{PRE_TRAINING_NAME}"
    torch.save(model.state_dict(), model_save_path)
    print("\nPre-training on %s procedure finished!" %PRE_TRAINING_NAME)
    #Change here the final layer depending on the model:
    model = change_model_top_layer(model, model_name)
else:
    print("\nNo pretraining was performed!")
    model_save_path = f"../../models/{MODEL_NAME}_untrained"
    torch.save(model.state_dict(), f"../../models/{MODEL_NAME}_untrained")


for rs in RANDOM_SEED:
    print("##############################")
    print(f"### Run {i}/{len(RANDOM_SEED)}, using random seed {rs}:")
    print("##############################")
    #reload model to initial values:
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
                  , pre_training_name=None\
                , pre_training=False  \
                , histories_path="." \
                , weight_decay=False)
    logits_df   = retrieve_logits(model        = model            \
                    , opt_df     = opt_df_train     \
                    , dataloader = dataloader_train \
                    , input_ids  = input_ids_train  \
                    , sentences  = sentences_train  \
                    , labels     = labels_train     \
                    , model_name = MODEL_NAME\
                    , batch_size = BATCH_SIZE\
                    , opt_data_path="."\
                    , pre_training_name=PRE_TRAINING_NAME)
