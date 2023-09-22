import os
import sys
import argparse
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import datetime
import math
from typing import Tuple
import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import stats
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import PreTrainedTokenizer
from transformers import pipeline
import matplotlib.pyplot as plt
import math
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from detection_malor import get_malor_heshe, get_malor_hisher, get_malor_malefemale
from detection_barchart import show_bar_charts

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='he_she or his_her or male_female', required=True)
    parser.add_argument('--sent', help='.csv file with sentence templates for bias evaluation', required=True)
    parser.add_argument('--occ', help='.csv file with occupations for bias evaluation', required=True)
    parser.add_argument('--dataset', help='.csv file with sentences for training', required=False)
    parser.add_argument('--detect', help=' none(0) or malor(1) or probability bar(2) or both(3)', required=False, default=3, type=int)
    parser.add_argument('--out', help='output directory + filename', required=True)
    parser.add_argument('--model', help='which BERT model to use', required=False)
    parser.add_argument('--batch', help='fix batch-size for fine-tuning', required=False, default=32, type=int)
    parser.add_argument('--epochs', help='fix epoch-size for fine-tuning', required=False, default=200, type=int)
    parser.add_argument('--seed', required=False, default=42, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    
    # creating output directory
    folder_name=args.out
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    beforeaftermalor_path=folder_name+"/beforeaftermalor.txt"
    mae_path=folder_name+"/malor.txt"
    epoch_path=folder_name+"/epoch.txt"
    plot_path=folder_name+"/plot.png"
    bar_before_gender_path=folder_name+"/bar_before_gender.png"
    bar_before_occ_path=folder_name+"/bar_before_occ.png"
    bar_after_gender_path=folder_name+"/bar_after_gender.png"
    bar_after_occ_path=folder_name+"/bar_after_occ.png"


    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('-- Importing model --')
    # determine which pre-trained BERT model to use:
    if args.model is None:
        print("Model chosen: bert-base-uncased")
        model_name='bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased',
                                            output_attentions=False,
                                            output_hidden_states=False)
    else:
        print("Model chosen:",args.model)
        model_name=args.model
        if (model_name=='roberta-base' or model_name=='roberta-large' or model_name=='albert-base-v1' or model_name=='albert-large-v1'):
            if (args.exp=='male_female'):
                print("The model is not suited for male names-female names experiment due to its lacking of the names in its vocabulary")
                sys.exit()
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                model = AutoModelForMaskedLM.from_pretrained(args.model,
                                                    output_attentions=False,
                                                    output_hidden_states=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForMaskedLM.from_pretrained(args.model,
                                                    output_attentions=False,
                                                    output_hidden_states=False)

    #setting templates and occupations
    sent_templates = pd.read_csv(args.sent)
    sentences = []
    for text in sent_templates.templates:
        sentences += sent_tokenize(text)
        
    occ_words = pd.read_csv(args.occ)
    occupations = []
    for text in occ_words.occupations:
        occupations += sent_tokenize(text)
    
    if (args.detect!=0):
        print('-- Creating Probability Bar Charts before training --')
        if (args.exp=="he_she" and (args.detect==2 or args.detect==3)):
            show_bar_charts(model, model_name, tokenizer, sentences, occupations, bar_before_gender_path, 1, 0, args.exp)
            if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'):
                show_bar_charts(model, model_name, tokenizer, sentences, occupations, bar_before_occ_path, 0, 1, args.exp)
            else:
                print("Probability bar chart not suited for this model due to its lacking of the occupations in its vocabulary")
                
        elif (args.exp=="his_her" and (args.detect==2 or args.detect==3)): 
            show_bar_charts(model, model_name, tokenizer, sentences, occupations, bar_before_gender_path, 1, 0, args.exp)
            if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'):
                show_bar_charts(model, model_name, tokenizer, sentences, occupations, bar_before_occ_path, 0, 1, args.exp)
            else:
                print("Probability bar chart not suited for this model due to its lacking of the occupations in its vocabulary")
                
        elif (args.exp=="male_female" and (args.detect==2 or args.detect==3)):
            if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'): 
                show_bar_charts(model, model_name, tokenizer, sentences, occupations, bar_before_gender_path, 1, 0, args.exp)
            else:
                print("Probability bar chart not suited for this model due to its lacking of the names in its vocabulary")
            if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'):
                show_bar_charts(model, model_name, tokenizer, sentences, occupations, bar_before_occ_path, 0, 1, args.exp)
            else:
                print("Probability bar chart not suited for this model due to its lacking of the occupations in its vocabulary")
        elif (args.detect==1):
            print("Not creating Probability Bar Charts")
        else:
            print("Experiments other than he_she, his_her, male_female are not supported yet")
            sys.exit()

    if (args.detect!=0):
        print('-- Calculating MALoR before training --')
        if (args.exp=="he_she" and (args.detect==1 or args.detect==3)):
            malor=get_malor_heshe(model,tokenizer,sentences,occupations)
            print("MALoR Before : ",malor)
            with open(beforeaftermalor_path, "a") as file:
                file.write("MALoR Before : " + str(malor) + "\n")
        elif (args.exp=="his_her" and (args.detect==1 or args.detect==3)):
            malor=get_malor_hisher(model,tokenizer,sentences,occupations)
            print("MALoR Before : ",malor)
            with open(beforeaftermalor_path, "a") as file:
                file.write("MALoR Before : " + str(malor) + "\n")
        elif (args.exp=="male_female" and (args.detect==1 or args.detect==3)):
            malor=get_malor_malefemale(model,tokenizer,sentences,occupations)
            print("MALoR Before : ",malor)
            with open(beforeaftermalor_path, "a") as file:
                file.write("MALoR Before : " + str(malor) + "\n")
        elif (args.detect==2):
            print("Not calculating MALoR")
        else:
            print("Experiments other than he_she, his_her, male_female are not supported yet")
            sys.exit()
        
    print('-- Continuing Training --')
    
    def plot_graph(malor_values, epoch_values, path, model_name, exp):
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Create a new figure and plot the learning curve.
        plt.figure()  
        plt.plot(malor_values, epoch_values, 'b-o')

        # Label the plot.
        plt.xlabel("Epoch")
        plt.ylabel("MALoR Score")
        
        # Add a title to the plot.
        plt.title(f"Model: {model_name}, Exp: {exp}")
        
        plt.savefig(path)
    
    def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
                "--mlm flag if you want to use this tokenizer. "
            )

        labels = inputs.clone()
        labels = labels.to(torch.long)

        probability_matrix = torch.full(labels.shape, mlm_probability) #decides which tokens to mask during masked language modeling
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist() #determine if each token is a special token [CLS],[SEP]
        ]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0) #updates the probability_matrix by setting the probabilities to 0.0 for tokens that are marked as special tokens 
        if tokenizer._pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id) #creates a padding_mask by checking if each token in labels is equal to the padding token ID
            probability_matrix.masked_fill_(padding_mask, value=0.0) #updates the probability_matrix by setting the probabilities to 0.0 for tokens that are part of padding
            
        masked_indices = torch.bernoulli(probability_matrix).bool() #binary mask called masked_indices which determines which tokens in the input should be masked (1 for masked, 0 for not masked)
        labels[~masked_indices] = -100  # tokens in the labels tensor that are not selected for masking (based on ~masked_indices) are set to -100 to indicate that these tokens should not contribute to the loss computation

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices #binary mask called indices_replaced created with a probability of 0.8, but only for the tokens that are selected for masking 
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) #replaces the tokens in the inputs tensor with the token ID corresponding to the [MASK]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced #binary mask named indices_random created with a probability of 0.5 and applies to tokens that are selected for masking but were not previously replaced 
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)  #contains random integer values between 0 and the length of the tokenizer's vocabulary
        inputs = inputs.to(torch.long)  
        random_words = random_words.to(torch.long)  
        inputs[indices_random] = random_words[indices_random] #replaces the masked tokens in the inputs tensor (determined by indices_random) with the corresponding random words from the random_words tensor

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def attention_mask_creator(input_ids): #creates attention mask list of lists: 0 only for [PAD] tokens. Performs for tokens in all sentences
        attention_masks = []
        for sent in input_ids:
            segments_ids = [int(t > 0) for t in sent]
            attention_masks.append(segments_ids)
        return torch.tensor(attention_masks)

    def tokenize_to_id(sentences, tokenizer): #tokenize the sentence, add [CLS], [SEP] and map tokens into IDs based on vocabulary. Performs for all senteces
        input_ids = []
        for sent in sentences:
            encoded_sent = tokenizer.encode(sent,add_special_tokens=True)
            input_ids.append(encoded_sent)
        return input_ids

    def input_pipeline(sequence, tokenizer, MAX_LEN): #takes tokenized id, add padding till MAX_LEN and create attention mask
        input_ids = tokenize_to_id(sequence, tokenizer)
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                                dtype="long",
                                value=tokenizer.mask_token_id,
                                truncating="post", padding="post")
        input_ids = torch.tensor(input_ids)
        attention_masks = attention_mask_creator(input_ids)
        return input_ids, attention_masks

    def format_time(elapsed): #takes a time in seconds and returns a string hh:mm:ss
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    malor_values=[]
    epoch_values =[]
    def fine_tune(model, dataloader, epochs, tokenizer, device):
        model.to(device)
        model.train()

        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)  

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        
        for epoch_i in range(0, epochs):
            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # measure how long the training epoch takes
            t0 = time.time()
            # reset the total loss for this epoch
            total_loss = 0
            for step, batch in enumerate(dataloader):
                # progress update every 40 batches
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # report progress
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))
                # mask inputs so the model can actually learn something
                b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
                b_input_ids = b_input_ids.to(device)
                b_labels = b_labels.to(device)
                b_input_mask = batch[1].to(device)
                model.zero_grad()
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs[0]
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the 'exploding gradients' problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            print('')
            #print('  Average training loss: {0:.2f}'.format(avg_train_loss))
            print('  Training epoch took: {:}'.format(format_time(time.time() - t0)))

            if (args.exp=="he_she"):
                malor=get_malor_heshe(model,tokenizer,sentences,occupations)
            elif (args.exp=="his_her"):
                malor=get_malor_hisher(model,tokenizer,sentences,occupations)
            elif (args.exp=="male_female"):
                malor=get_malor_malefemale(model,tokenizer,sentences,occupations)

            global malor_values
            global epoch_values
            epoch_values.append(epoch_i+1)
            malor_values.append(malor)
        print('Training complete!')
        return model
    
    if args.dataset:
        tune_corpus = pd.read_csv(args.dataset)
        tune_data = []
        for text in tune_corpus.occupation_sentences:
            tune_data += sent_tokenize(text)
     
        # as max_len get the smallest power of 2 greater or equal to the max sentence length
        max_len_tune = max([len(sent.split()) for sent in tune_data])
        pos = math.ceil(math.log2(max_len_tune))
        max_len_tune = int(math.pow(2, pos))
    
        # get tokens and attentions tensor for training the data
        tune_tokens, tune_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
        assert tune_tokens.shape == tune_attentions.shape

        # set up Dataloader
        batch_size = int(args.batch)
        train_data = TensorDataset(tune_tokens, tune_attentions)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        
        print('-- Setting up model training --')
        epochs = args.epochs
        model = fine_tune(model, train_dataloader, epochs, tokenizer, device)

        model.save_pretrained(folder_name)                  
        tokenizer.save_pretrained(folder_name)
        
        with open(mae_path, 'w') as file:
            file.write(str(malor_values))
            file.close()

        with open(epoch_path, 'w') as file:
            file.write(str(epoch_values))
            file.close()
        
        plot_graph(epoch_values, malor_values, plot_path, model_name, args.exp)
        
        model = AutoModelForMaskedLM.from_pretrained(folder_name)
        tokenizer = AutoTokenizer.from_pretrained(folder_name)
        new_model_name = "debiased" + model_name

        if (args.detect!=0):        
            print('-- Creating Probability Bar Charts after training --')
            if (args.exp=="he_she" and (args.detect==2 or args.detect==3)):
                show_bar_charts(model, new_model_name, tokenizer, sentences, occupations, bar_after_gender_path, 1, 0, args.exp)
                if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'):
                    show_bar_charts(model, new_model_name, tokenizer, sentences, occupations, bar_after_occ_path, 0, 1, args.exp)
                else:
                    print("Probability bar chart not suited for this model due to its lacking of the occupations in its vocabulary")
                    
            elif (args.exp=="his_her" and (args.detect==2 or args.detect==3)): 
                show_bar_charts(model, new_model_name, tokenizer, sentences, occupations, bar_after_gender_path, 1, 0, args.exp)
                if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'):
                    show_bar_charts(model, new_model_name, tokenizer, sentences, occupations, bar_after_occ_path, 0, 1, args.exp)
                else:
                    print("Probability bar chart not suited for this model due to its lacking of the occupations in its vocabulary")
                    
            elif (args.exp=="male_female" and (args.detect==2 or args.detect==3)):
                if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'): 
                    show_bar_charts(model, new_model_name, tokenizer, sentences, occupations, bar_after_gender_path, 1, 0, args.exp)
                else:
                    print("Probability bar chart not suited for this model due to its lacking of the names in its vocabulary")
                if (model_name=='bert-base-uncased' or model_name=='bert-large-uncased' or model_name=='distilbert-base-uncased'):
                    show_bar_charts(model, new_model_name, tokenizer, sentences, occupations, bar_after_occ_path, 0, 1, args.exp)
                else:
                    print("Probability bar chart not suited for this model due to its lacking of the occupations in its vocabulary")
            elif (args.detect==1):
                print("Not creating Probability Bar Charts")

        if (args.detect!=0):  
            print('-- Calculating MALoR after training --')
            if (args.exp=="he_she" and (args.detect==1 or args.detect==3)):
                malor=get_malor_heshe(model,tokenizer,sentences,occupations)
                print("MALoR After : ",malor)
                with open(beforeaftermalor_path, "a") as file:
                    file.write("MALoR After : " + str(malor))
            elif (args.exp=="his_her" and (args.detect==1 or args.detect==3)):
                malor=get_malor_hisher(model,tokenizer,sentences,occupations)
                print("MALoR After : ",malor)
                with open(beforeaftermalor_path, "a") as file:
                    file.write("MALoR After : " + str(malor))
            elif (args.exp=="male_female" and (args.detect==1 or args.detect==3)):
                malor=get_malor_malefemale(model,tokenizer,sentences,occupations)
                print("MALoR After : ",malor)
                with open(beforeaftermalor_path, "a") as file:
                    file.write("MALoR After : " + str(malor))
            elif (args.detect==2):
                print("Not calculating MALoR")

        else:
            print('Only Detection, No Training')


