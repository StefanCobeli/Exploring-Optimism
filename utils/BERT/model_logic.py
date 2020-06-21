import datetime
import numpy     as np
import pandas    as pd
import random
import time
import torch

from sklearn.metrics import f1_score
from transformers    import AlbertForSequenceClassification, BertForSequenceClassification
from transformers    import RobertaForSequenceClassification, XLNetForSequenceClassification
from tqdm            import tqdm
#Model that are considered in the paper.
#Further models can obviously be added:
# MODEL_NAME = ['albert-base-v2'\
#               , 'bert-base-uncased', 'bert-large-uncased'\
#               , 'roberta-base', 'xlnet-base-cased',  ]
print("Loaded model_logic module!")

def change_model_top_layer(model, model_name):
    """
        Change models' top layer based on its architecture.
        Returns the model with only 2 output units.
    """
    if model_name=='albert-base-v2':
        model.classifier = torch.nn.Linear(in_features=768\
                                        , out_features=2)
    if model_name=='bert-base-uncased':
        model.classifier = torch.nn.Linear(in_features=768\
                                        , out_features=2)
    if model_name=='bert-large-uncased':
        model.classifier = torch.nn.Linear(in_features=1024\
                                        , out_features=2)
    if model_name=="roberta-base":
        model.out_proj = torch.nn.Linear(in_features=768\
                                      , out_features=2)
    if model_name=="xlnet-base-cased":
        model.logits_proj = torch.nn.Linear(in_features=768\
                                         , out_features=2)
    return model

def pick_model(model_name, num_labels):
    """
        Return specified model:
        Available model names:
        ['albert-base-v2'\
          , 'bert-base-uncased', 'bert-large-uncased'\
          , 'roberta-base', 'xlnet-base-cased',  ]
    """
    if model_name == 'albert-base-v2':
        model = AlbertForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    if model_name in ('bert-base-uncased', 'bert-large-uncased'):
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    if model_name == 'roberta-base':
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    if model_name == 'xlnet-base-cased':
        model = XLNetForSequenceClassification.from_pretrained(
            model_name,
            num_labels = num_labels,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

    print(f'Loaded {model_name} model.')
    if torch.cuda.is_available():
        model.cuda()
        
    return model



# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels, F1=False):
    if F1:
        return f1_score(preds.argmax(axis=1), labels, average="weighted")
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Get predicted logits by model on dataloader:
def retrieve_logits(model, opt_df          \
                    , dataloader, input_ids\
                    , sentences, labels    \
                    , model_name, batch_size  \
                    , opt_data_path, pre_training_name=None\
                    , iteration="1of1"
                    , data_type="train"
                    , setting="set0"):
    '''
        Return data_frame with additional 2 columns according to
        the 2 logits predicted by the model for each entry.
    '''

    #Criterion:
    cse_loss = torch.nn.CrossEntropyLoss()
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Tracking variables
    total_accuracy = 0
    total_loss = 0
    nb_steps = 0
    #Test if tweets are weel retrieved:
    assertion_tweets  = [None for i in labels]
    #Store predicted logits (the first & second):
    predicted_logits0 = [None for i in labels]
    predicted_logits1 = [None for i in labels]

    t0=time.time()
    # Evaluate data for one epoch
    for batch in dataloader:
        b_input_ids  = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels     = batch[2].to(device)
        with torch.no_grad():
            logits = model(b_input_ids,
               token_type_ids=None,
               attention_mask=b_input_mask)[0]
            loss   = cse_loss(logits, b_labels)
        # Accumulate the validation loss.
        total_loss += loss.item()* (len(logits)/batch_size)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        for i, sample in enumerate(batch[0]):
            original_index = input_ids.tolist().index(batch[0][i].tolist())

            assert(sentences[original_index] ==  opt_df.iloc[original_index]["Tweet"])
            predicted_logits0[original_index] = logits[i][0]
            predicted_logits1[original_index] = logits[i][1]
            assertion_tweets[original_index]  = sentences[original_index]
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_accuracy += flat_accuracy(logits, label_ids) * (len(logits)/batch_size)

    # Report the final accuracy for this validation run.
    avg_accuracy = total_accuracy / len(dataloader)
    print("\n Accuracy of model {0} on {1}: {2:.4f}".format(model_name, data_type, avg_accuracy))
    # Calculate the average loss over all of the batches.
    avg_loss = total_loss / len(dataloader)
    # Measure how long the validation run took.
    computation_time = format_time(time.time() - t0)
    print(" Loss: {0:.2f}".format(avg_loss))
    print(" Computation took: {:}".format(computation_time))


    opt_df[model_name+"logit0"] = predicted_logits0
    opt_df[model_name+"logit1"] = predicted_logits1
#     assert(np.where(mock_opt_df["Tweet"] == mock_opt_df["Model1_tweet"])[0].shape[0] == mock_opt_df.shape[0])
    assert(np.all(np.where(opt_df["Tweet"] == assertion_tweets, True, False)))

    #Save new dataframe:
    data_name = f"OPT_{pre_training_name+'_' if pre_training_name else ''}"
    save_path = f"{opt_data_path}Logits/{model_name}/Logits_{data_name}{model_name}_{setting}_it:{iteration}_{data_type}_Acc:{avg_accuracy}.csv"
    opt_df.to_csv(save_path)

    return opt_df


#Training loop.
def train_model(model, optimizer\
                , batch_size, dataloader_train     \
                , dataloader_test, dataloader_val  \
                , num_epochs, random_seed\
                , model_name, pre_training_name=None\
                , pre_training=False  \
                , iteration=""
                , histories_path="." \
                , weight_decay=False
                , setting="set0"):
    """
        Train the the given `model` using `optimizer`, `batch_size`
        and the provided Dataloaders, for `num_epochs` with `random_seed`.
        model_name:        should be a cosntant (`MODEL_NAME`)
            in the program (see `pick_model`);
        pre_training_name: if the model was/is on pre-training
            (`PRE_TRAINING_NAME` constant) ;
        pre_training:      True of False if pre-training data;
        histories_path:    Where to save training stats;
        weight_decay:      optimizer decay weights (untested feature).

        Returns: Training stats as pandas DataFrame.
    """

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    # and on:
    # https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
    # Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    if weight_decay:
        # Create the learning rate scheduler.
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                            num_warmup_steps = 0, # Default value in run_glue.py
                            num_training_steps = len(dataloader_train) * num_epochs)

    #Criterion:
    cse_loss = torch.nn.CrossEntropyLoss()
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # For each epoch...
    for epoch_i in (range(num_epochs)):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss     = 0
        total_train_accuracy = 0

        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(dataloader_train)):
            if not(torch.cuda.is_available()) and step % 120 == 1:
                #If there are not many resources, just test for runtime errors:
                break

            # Unpack this training batch from our dataloader.
            b_input_ids  = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels     = batch[2].to(device)
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
#             loss, logits = model1(b_input_ids,
#                                  token_type_ids=None,
#                                  attention_mask=b_input_mask,
#                                  labels=b_labels)
            logits = model(b_input_ids,
               token_type_ids=None,
               attention_mask=b_input_mask)[0]
            loss   = cse_loss(logits, b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item() * (len(logits)/batch_size)

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            if weight_decay:
                # Update the learning rate.
                scheduler.step()

            # Move logits and labels to CPU
            logits    = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy += flat_accuracy(logits, label_ids, F1=False) * (len(logits)/batch_size)


        # Report the final accuracy for this validation run.
        avg_train_accuracy = total_train_accuracy / len(dataloader_train)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(dataloader_train)
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print(f"  Average training Accuracy {model_name}: {avg_train_accuracy:.4f}")
        print(f"  Average training loss {model_name}: {avg_train_loss:.2f}")
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.


        ###################################################
        #############Validation Stats######################
        ###################################################
        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss     = 0

        # Evaluate data for one epoch
        for batch in dataloader_val:
            b_input_ids  = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels     = batch[2].to(device)
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                logits = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask)[0]
                loss   = cse_loss(logits, b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item() * (len(logits)/batch_size)

            # Move logits and labels to CPU
            logits     = logits.detach().cpu().numpy()
            label_ids  = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids, F1=False) * (len(logits)/batch_size)

            if not(torch.cuda.is_available()):
                break


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(dataloader_val)
        print(f"  Validation Accuracy {model_name}: {avg_val_accuracy:.4f}")

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dataloader_val)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print(f"  Validation Loss {model_name}: {avg_val_loss:.2f}")
        print("  Validation took: {:}".format(validation_time))


        ###################################################
        #############Test Stats############################
        ###################################################
        print("")
        print("Running Test...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_test_accuracy = 0
        total_test_loss     = 0
        # Evaluate data for one epoch
        for batch in dataloader_test:
            b_input_ids  = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels     = batch[2].to(device)
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                logits = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask)[0]
                loss   = cse_loss(logits, b_labels)
            # Accumulate the validation loss.
            total_test_loss += loss.item() * (len(logits)/batch_size)
            # Move logits and labels to CPU
            logits    = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_test_accuracy += flat_accuracy(logits, label_ids, F1=False) * (len(logits)/batch_size)
            if not(torch.cuda.is_available()):
                break


        # Report the final accuracy for this validation run.
        avg_test_accuracy = total_test_accuracy / len(dataloader_test)
        print(f"  Test Accuracy {model_name}: {avg_test_accuracy:.4f}")

        # Calculate the average loss over all of the batches.
        avg_test_loss = total_test_loss / len(dataloader_test)

        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)

        print(f"  Test Loss {model_name}: {avg_test_loss:.2f}")
        print("  Test took: {:}".format(test_time))


        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                f'Training Loss': avg_train_loss,
                f'Valid. Loss': avg_val_loss,
                f'Test Loss': avg_test_loss,
                f'Training Accur.': avg_train_accuracy,
                f'Valid. Accur.': avg_val_accuracy,
                f'Test Accur.': avg_test_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'Test Time': test_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    #############################
    #Saving training stats:
    df_stats  = pd.DataFrame(data=training_stats).set_index('epoch')
    val_acc   = df_stats["Valid. Accur."].values[-1]#.max()
    data_name = pre_training_name + "_" if pre_training \
            else f"OPT_{pre_training_name+'_' if pre_training_name else ''}"
    hist_fn = f"{histories_path}{data_name}{model_name}_{setting}_it:{iteration}_ValAcc:{val_acc}.csv"
    df_stats.to_csv(hist_fn)
    print(f"\nTraining stats saved at: {hist_fn}.")

    return model, df_stats
