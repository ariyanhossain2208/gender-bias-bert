from transformers import pipeline
import torch
import matplotlib.pyplot as plt
import math
from pandas import *
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from tqdm import tqdm
import pandas as pd
from nltk import sent_tokenize
import os


def show_bar_charts(model, model_name, tokenizer, sentences, occupations, path, gender, occ, exp):
    bar_width = 0.4
    selected_occupations = ['engineer', 'librarian', 'nurse', 'surgeon', 'programmer', 'chef', 'scientist', 'secretary', 'architect', 'teacher']
    
    if (gender == 1 and occ == 0):
        occupation_scores = get_occupation_average_gender(model, tokenizer, sentences, selected_occupations, exp)
    if (gender == 0 and occ == 1):
        occupation_scores = get_occupation_average_occ(model, tokenizer, sentences, selected_occupations, exp)
         
    if (exp == "he_she"):   
        he_scores = [occupation_scores[j][0] for j in selected_occupations]
        she_scores = [occupation_scores[j][1] for j in selected_occupations]
        he_norm = [(he / (he + she)) for he, she in zip(he_scores, she_scores)]
        she_norm = [(she / (he + she)) for he, she in zip(he_scores, she_scores)]
        fig, ax = plt.subplots(figsize=(12, 6))
        if (gender == 1):
            fig.suptitle("Model: {}, Masking: Gender".format(model_name), y=0.95)
        if (occ == 1):
            fig.suptitle("Model: {}, Masking: Occupation".format(model_name), y=0.95)
        x = np.arange(len(selected_occupations))
        rects1 = ax.bar(x - bar_width/2, he_norm, bar_width, label="he")
        rects2 = ax.bar(x + bar_width/2, she_norm, bar_width, label="she")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_occupations)
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        plt.savefig(path)
        
    elif (exp == "his_her"):   
        his_scores = [occupation_scores[j][0] for j in selected_occupations]
        her_scores = [occupation_scores[j][1] for j in selected_occupations]
        his_norm = [(his / (his + her)) for his, her in zip(his_scores, her_scores)]
        her_norm = [(her / (his + her)) for his, her in zip(his_scores, her_scores)]
        fig, ax = plt.subplots(figsize=(12, 6))
        if (gender == 1):
            fig.suptitle("Model: {}, Masking: Gender".format(model_name), y=0.95)
        if (occ == 1):
            fig.suptitle("Model: {}, Masking: Occupation".format(model_name), y=0.95)
        x = np.arange(len(selected_occupations))
        rects1 = ax.bar(x - bar_width/2, his_norm, bar_width, label="his", color="indianred")
        rects2 = ax.bar(x + bar_width/2, her_norm, bar_width, label="her", color="slateblue")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_occupations)
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        plt.savefig(path)     
        
    elif (exp == "male_female"): 
        malename_scores = [occupation_scores[j][0] for j in selected_occupations]
        femalename_scores = [occupation_scores[j][1] for j in selected_occupations]
        male_norm = [(male / (male + female)) for male, female in zip(malename_scores, femalename_scores)]
        female_norm = [(female / (male + female)) for male, female in zip(malename_scores, femalename_scores)]
        fig, ax = plt.subplots(figsize=(12, 6))
        if (gender == 1):
            fig.suptitle("Model: {}, Masking: Gender".format(model_name), y=0.95)
        if (occ == 1):
            fig.suptitle("Model: {}, Masking: Occupation".format(model_name), y=0.95)
        x = np.arange(len(selected_occupations))
        rects1 = ax.bar(x - bar_width/2, male_norm, bar_width, label="male names", color="chocolate")
        rects2 = ax.bar(x + bar_width/2, female_norm, bar_width, label="female names", color="gold")
        ax.set_xticks(x)
        ax.set_xticklabels(selected_occupations)
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        plt.savefig(path) 
        
    else:
        print("Experiments other than he_she, his_her, male_female are not supported yet")    
        
        
#Masking Gender Term    
def get_occupation_average_gender(model, tokenizer, sentences, occupations, exp):
    occupation_scores = {occupation: [] for occupation in occupations}
    nlp = pipeline("fill-mask",model = model,tokenizer = tokenizer,device=0 if torch.cuda.is_available() else -1)
    
    if (exp == "he_she"):
        for occupation in tqdm(occupations):
            male_scores = []
            female_scores = []
            for sentence in sentences:
                sentence_structure = sentence.replace("[MASK]",f"{nlp.tokenizer.mask_token}")
                sentence_structure = sentence_structure.replace("[OCC]",occupation)
                output = nlp(sentence_structure, targets=['he', 'she'])
                if output[0]['token_str'] == "he":
                    male_score = output[0]['score']
                    female_score = output[1]['score']
                else:
                    male_score = output[1]['score']
                    female_score = output[0]['score']
                male_scores.append(male_score)
                female_scores.append(female_score)
            male_avg = sum(male_scores) / len(male_scores)
            female_avg = sum(female_scores) / len(female_scores)
            occupation_scores[occupation].extend([male_avg, female_avg])
        return occupation_scores
    
    elif (exp == "his_her"):
        for occupation in tqdm(occupations):
            male_scores = []
            female_scores = []
            for sentence in sentences:
                sentence_structure = sentence.replace("[MASK]",f"{nlp.tokenizer.mask_token}")
                sentence_structure = sentence_structure.replace("[OCC]",occupation)
                output = nlp(sentence_structure, targets=['his', 'her'])
                if output[0]['token_str'] == "his":
                    male_score = output[0]['score']
                    female_score = output[1]['score']
                else:
                    male_score = output[1]['score']
                    female_score = output[0]['score']
                male_scores.append(male_score)
                female_scores.append(female_score)
            male_avg = sum(male_scores) / len(male_scores)
            female_avg = sum(female_scores) / len(female_scores)
            occupation_scores[occupation].extend([male_avg, female_avg])
        return occupation_scores  
     
    elif (exp == "male_female"):
        male_names = ['Michael', 'David', 'James', 'John', 'Robert', 'William', 'Richard', 'Thomas', 'Christopher', 'Joseph', 'Steven', 'Paul', 'Daniel', 'Andrew', 'Kenneth', 'George', 'Charles', 'Stephen', 'Anthony', 'Edward', 'Brian', 'Ronald', 'Kevin', 'Matthew', 'Jason', 'Timothy', 'Gary', 'Jeffrey', 'Scott']
        female_names = ['Jennifer', 'Linda', 'Patricia', 'Susan', 'Mary', 'Sarah', 'Jessica', 'Elizabeth', 'Karen', 'Nancy', 'Lisa', 'Margaret', 'Betty', 'Sandra', 'Ashley', 'Dorothy', 'Kimberly', 'Emily', 'Michelle', 'Laura', 'Rebecca', 'Amanda', 'Carol', 'Helen', 'Sharon', 'Cynthia', 'Kathleen', 'Amy', 'Melissa']
        male_names = [name.lower() for name in male_names]
        female_names = [name.lower() for name in female_names]
        all_names = male_names + female_names
        for occupation in tqdm(occupations):
            male_score = []
            female_score = []
            for sentence in sentences:
                sentence_structure = sentence.replace("[MASK]",f"{nlp.tokenizer.mask_token}")
                sentence_structure = sentence_structure.replace("[OCC]",occupation)
                output = nlp(sentence_structure, targets=all_names, top_k=58)
                male_prob = [item['score'] for item in output if item['token_str'] in male_names]
                female_prob = [item['score'] for item in output if item['token_str'] in female_names]
                male_prob_avg = sum(male_prob) / len(male_prob)
                female_prob_avg = sum(female_prob) / len(female_prob)
                male_score.append(male_prob_avg)
                female_score.append(female_prob_avg)
            male_avg = sum(male_score) / len(male_score)
            female_avg = sum(female_score) / len(female_score)
            occupation_scores[occupation].append(male_avg)
            occupation_scores[occupation].append(female_avg)
        return occupation_scores  
    
    else:
        print("Experiments other than he_she, his_her, male_female are not supported yet")       

#Masking Occupation Term
def get_occupation_average_occ(model, tokenizer, sentences, occupations, exp):
    occupation_scores = {occupation: [] for occupation in occupations}
    nlp = pipeline("fill-mask", model = model, tokenizer = tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    if (exp == "he_she"):
        for occupation in tqdm(occupations):
            male_scores = []
            female_scores = []
            for sentence in sentences:
                sentence_structure_he = sentence.replace("[MASK]","he")
                sentence_structure_he = sentence_structure_he.replace("[OCC]",f"{nlp.tokenizer.mask_token}")
                sentence_structure_she = sentence.replace("[MASK]","she")
                sentence_structure_she = sentence_structure_she.replace("[OCC]",f"{nlp.tokenizer.mask_token}")
                output_he = nlp(sentence_structure_he, targets=[occupation])
                output_she = nlp(sentence_structure_she, targets=[occupation])
                score_he = output_he[0]['score']
                token_he = output_he[0]['token_str']
                score_she = output_she[0]['score']
                token_she = output_she[0]['token_str']
                if token_he == occupation:
                    male_scores.append(score_he)
                if token_she == occupation:
                    female_scores.append(score_she)
            male_avg = sum(male_scores) / len(male_scores)
            female_avg = sum(female_scores) / len(female_scores)
            occupation_scores[occupation].extend([male_avg, female_avg])
        return occupation_scores
    
    elif (exp == "his_her"):
        for occupation in tqdm(occupations):
            male_scores = []
            female_scores = []
            for sentence in sentences:
                sentence_structure_his = sentence.replace("[MASK]","his")
                sentence_structure_his = sentence_structure_his.replace("[OCC]",f"{nlp.tokenizer.mask_token}")
                sentence_structure_her = sentence.replace("[MASK]","her")
                sentence_structure_her = sentence_structure_her.replace("[OCC]",f"{nlp.tokenizer.mask_token}")
                output_his = nlp(sentence_structure_his, targets=[occupation])
                output_her = nlp(sentence_structure_her, targets=[occupation])
                score_his = output_his[0]['score']
                token_his = output_his[0]['token_str']
                score_her = output_her[0]['score']
                token_her = output_her[0]['token_str']
                if token_his == occupation:
                    male_scores.append(score_his)
                if token_her == occupation:
                    female_scores.append(score_her)
            male_avg = sum(male_scores) / len(male_scores)
            female_avg = sum(female_scores) / len(female_scores)
            occupation_scores[occupation].extend([male_avg, female_avg])
        return occupation_scores
    
    elif (exp == "male_female"):
        male_names = ['Michael', 'David', 'James', 'John', 'Robert', 'William', 'Richard', 'Thomas', 'Christopher', 'Joseph', 'Steven', 'Paul', 'Daniel', 'Andrew', 'Kenneth', 'George', 'Charles', 'Stephen', 'Anthony', 'Edward', 'Brian', 'Ronald', 'Kevin', 'Matthew', 'Jason', 'Timothy', 'Gary', 'Jeffrey', 'Scott']
        female_names = ['Jennifer', 'Linda', 'Patricia', 'Susan', 'Mary', 'Sarah', 'Jessica', 'Elizabeth', 'Karen', 'Nancy', 'Lisa', 'Margaret', 'Betty', 'Sandra', 'Ashley', 'Dorothy', 'Kimberly', 'Emily', 'Michelle', 'Laura', 'Rebecca', 'Amanda', 'Carol', 'Helen', 'Sharon', 'Cynthia', 'Kathleen', 'Amy', 'Melissa']
        male_names = [name.lower() for name in male_names]
        female_names = [name.lower() for name in female_names]
        for occupation in tqdm(occupations):
            male_scores = []
            female_scores = []
            for sentence in sentences:
                male_prob = []
                for i in male_names:
                    sentence_structure = sentence.replace("[MASK]", i)
                    sentence_structure = sentence_structure.replace("[OCC]",f"{nlp.tokenizer.mask_token}")
                    output = nlp(sentence_structure, targets=[occupation])
                    male_prob.append(output[0]['score'])
                male_prob_avg = sum(male_prob) / len(male_prob)
                male_scores.append(male_prob_avg)
                
                female_prob = []
                for i in female_names:
                    sentence_structure = sentence.replace("[MASK]", i)
                    sentence_structure = sentence_structure.replace("[OCC]",f"{nlp.tokenizer.mask_token}")
                    output = nlp(sentence_structure, targets=[occupation])
                    female_prob.append(output[0]['score'])
                female_prob_avg = sum(female_prob) / len(female_prob)
                female_scores.append(female_prob_avg)
                
            male_avg = sum(male_scores) / len(male_scores)
            female_avg = sum(female_scores) / len(female_scores)
            occupation_scores[occupation].append(male_avg)
            occupation_scores[occupation].append(female_avg)
        return occupation_scores

    else:
        print("Experiments other than he_she, his_her, male_female are not supported yet")
          
 