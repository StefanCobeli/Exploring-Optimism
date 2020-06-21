import torch

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset, SequentialSampler
from transformers     import AlbertTokenizer, BertTokenizer
from transformers     import RobertaTokenizer, XLNetTokenizer
from tqdm import tqdm



#Model that are considered in the paper.
#Further models can obviously be added:
# MODEL_NAME = ['albert-base-v2'\
#               , 'bert-base-uncased', 'bert-large-uncased'\
#               , 'roberta-base', 'xlnet-base-cased',  ]
print("Loaded tokenization module!")

def pick_tokenizer(model_name='albert-base-v2'):
    """
        Return specified tokenizer:
        Available model names:
        ['albert-base-v2'\
          , 'bert-base-uncased', 'bert-large-uncased'\
          , 'roberta-base', 'xlnet-base-cased',  ]
    """
    if model_name == 'albert-base-v2':
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_name == 'bert-large-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_name == 'xlnet-base-cased':
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=True)

    print(f'Loaded {model_name} tokenizer.')
    return tokenizer

#Data tokenization:
def retrieve_data_encodings(sentences, tokenizer, max_len=64):
    """
        Returns input_ids & attentin_masks
        out of list of sentences (data tokenization).
    """
    input_ids       = []
    attention_masks = []

    # For every sentence...
    for sent in tqdm(sentences):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        # The method`encode` returns only the phrase encoding,
        # not the masks as well.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # input_ids       = torch.array(input_ids)
    # attention_masks = torch.array(attention_masks)
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def retrieve_dataloaders(ids_train, masks_train, labels_train \
                         , ids_test, masks_test, labels_test  \
                         , ids_val, masks_val, labels_val     \
                         , batch_size=32):
    """
        Transform lists of inputs into dataloaders.
        Returns:
        dataloader_train, dataloader_test, dataloader_validation
    """

    # Combine the training inputs into a TensorDataset.
    dataset_train    = TensorDataset(ids_train, masks_train, labels_train)
    dataloader_train = DataLoader(
                dataset_train,  # The training samples.
                sampler = RandomSampler(dataset_train), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    dataset_test = TensorDataset(ids_test, masks_test, labels_test)
    dataloader_test = DataLoader(
            dataset_test, # The validation samples.
            sampler = SequentialSampler(dataset_test), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

    dataset_val = TensorDataset(ids_val, masks_val, labels_val)
    dataloader_validation = DataLoader(
            dataset_val, # The validation samples.
            sampler = SequentialSampler(dataset_val), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

    print("Dataloaders lengths are: Train: %d, Test: %d, Val.: %d.\n" \
          %(len(dataloader_train), len(dataloader_test), len(dataloader_validation)))
    return dataloader_train, dataloader_test, dataloader_validation
