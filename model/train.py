import pandas as pd
import torch
from tqdm import tqdm
from transformers import (BertTokenizer, 
                          BertForSequenceClassification, 
                          AdamW, 
                          BertConfig, 
                          get_linear_schedule_with_warmup)
from torch.utils.data import (TensorDataset, 
                              random_split, 
                              DataLoader, 
                              RandomSampler, 
                              SequentialSampler)
import numpy as np
import time
import datetime
import random
from typing import Tuple
import os
import sys

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.DEBUG)

import common # Helper functions and params


def get_BertTokenizer(pretrained_path: str = 'bert-base-uncased', do_lower_case: bool=True) -> BertTokenizer:
    r"""
    Fetch a tokenizer.
    If not path is given, default to bert-base-uncased.
    
    Args:
        pretrained_path (:obj:`string` defaults to :obj:`bert-base-uncased`):
            Path of the pretrained model
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
            
    Returns:
        tokenizer (:obj:`transformers.BertTokenizer`)
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=do_lower_case)
    
    return tokenizer


def ret_model(pretrained_path: str = 'bert-base-uncased', num_labels: int=2) -> BertForSequenceClassification:
    r"""
    Fetch a model.
    If not path is given, default to bert-base-uncased.
    
    Args:
        pretrained_path (:obj:`string` defaults to :obj:`bert-base-uncased`):
            Path of the pretrained model
        num_labels (:obj:`int`, defaults to :obj:`2`):
            Number of labels to incorporate into the final layer.
            
    Returns:
        model (:obj:`transformers.BertForSequenceClassification`)
    """
    model = BertForSequenceClassification.from_pretrained(
        pretrained_path, 
        num_labels = num_labels, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    return model


def create_tokenized_data_set(df: pd.DataFrame, tokenizer: BertTokenizer) -> TensorDataset:
    r"""
    Tokenize a dataframe.
    
    Args:
        df (:obj:`pd.DataFrame`):
            Dataframe. Must contain 2 columns, text and label.
        tokenizer (:obj:`transformers.BertTokenizer`):
            Tokenizer to use.
            
    Returns:
        TensorDataset (:obj:`torch.utils.data.TensorDataset`)
    """
    
    # Load the BERT tokenizer.
    transformers_logger.info('Loading BERT tokenizer...')
    '''
    Original:  Our friends won't buy this analysis, let alone the next one we propose.
    Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
    Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]
    '''

    texts = df.text.values
    labels = df.label.values
    
    
    # Get max setence length
    max_len = 0

    # For every sentence...
    for text in texts:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        # Update the maximum length.
        max_len = max(max_len, len(input_ids))

    transformers_logger.debug('Max text length: max_len=%s', max_len)
    if max_len > common.MAX_SEQUENCE_LENGTH:
        transformers_logger.warning('Max text length, max_len=%s, exceeds MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH=%S', max_len, MAX_SEQUENCE_LENGTH)
        
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for text in texts:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            text,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = common.MAX_SEQUENCE_LENGTH,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def ret_dataloader(train_df: TensorDataset, eval_df: TensorDataset):
    batch_size = common.BATCH_SIZE
    transformers_logger.debug('batch_size: batch_size=%s', batch_size)
    train_dataloader = DataLoader(
                train_df,  # The training samples.
                sampler = RandomSampler(train_df), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                eval_df, # The validation samples.
                sampler = SequentialSampler(eval_df), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader,validation_dataloader



def ret_optim(model):
    transformers_logger.debug('Learning_rate: common.LEARNING_RATE=%s', common.LEARNING_RATE)
    optimizer = AdamW(model.parameters(),
                      lr = common.LEARNING_RATE, 
                      eps = 1e-8 
                    )
    return optimizer


def ret_scheduler(train_dataloader, optimizer):
    transformers_logger.debug('Epochs: common.EPOCHS=%s', common.EPOCHS)
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * common.EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    return scheduler


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
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


def split_train_test(dataset: TensorDataset, split_ratio: float = 0.7) ->Tuple[TensorDataset,TensorDataset]:

    # Calculate the number of samples to include in each set.
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def train(train_df, eval_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ret_model()
    model.to(device)
    train_dataloader, validation_dataloader = ret_dataloader(train_df, eval_df)
    optimizer = ret_optim(model)
    scheduler = ret_scheduler(train_dataloader,optimizer)

    seed_val = 42
   
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    # For each epoch...
    for epoch_i in range(0, common.EPOCHS):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
     
        transformers_logger.info('Training')
        transformers_logger.info('======== Epoch %s / %s ======== ', epoch_i + 1, common.EPOCHS)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                transformers_logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

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
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            transformers_logger.info('train_batch_loss: %s', loss.item())
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        transformers_logger.info('Training time: %s', training_time)
        
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        transformers_logger.info('Training')

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids, 
                                      token_type_ids=None, 
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)        

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        transformers_logger.info('val_accuracy: %s', avg_val_accuracy)
        transformers_logger.info('avg_val_loss: %s', avg_val_loss)
        transformers_logger.info('Validation time: %s', validation_time)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    transformers_logger.info('Total time: %s', format_time(time.time()-total_t0))
    return model


if __name__ == "__main__":
   
    # Retrieve data and labels
#     raw_training = pd.read_csv(os.path.join(os.environ["SM_CHANNEL_TRAINING"], "testing-train.csv"))
    raw_training = pd.read_csv(os.environ["SM_CHANNEL_TRAINING"])

#     raw_training = pd.read_csv("/home/ec2-user/SageMaker/hugging_face_testing/data/train.csv")

    # Tokenize
    tokenizer = get_BertTokenizer()
    tokenized_data = create_tokenized_data_set(raw_training, tokenizer)
    
    # Train and eval split
    train_df, eval_df = split_train_test(tokenized_data)
        
    model = train(train_df, eval_df)
        
    model.save_pretrained(os.environ["SM_MODEL_DIR"])
    tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])
        
    transformers_logger.info('finished')
    sys.exit(0)
