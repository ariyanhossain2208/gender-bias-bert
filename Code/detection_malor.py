
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import math
from transformers import pipeline
import numpy as np
import torch
from tqdm import tqdm

#he-she
def get_occupation_average_heshe(model,tokenizer,sentences,occupations):
    occupation_scores = {occupation: [] for occupation in occupations}
    average_scores = {occupation: [] for occupation in occupations}
    nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    for occupation in tqdm(occupations):
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
            main_score = math.log2(male_score/female_score)
            occupation_scores[occupation].append(main_score)
    for occupation in occupation_scores:
        temp = occupation_scores[occupation]
        avg = sum(temp)/len(temp)
        average_scores[occupation].append(avg)
    return average_scores
def get_malor_heshe(model,tokenizer,sentences,occupations):
    avg_log_score = get_occupation_average_heshe(model,tokenizer,sentences,occupations)
    values = list(avg_log_score.values())
    return (np.mean(np.abs(values)))

#his-her
def get_occupation_average_hisher(model,tokenizer,sentences,occupations):
    occupation_scores = {occupation: [] for occupation in occupations}
    average_scores = {occupation: [] for occupation in occupations}
    nlp = pipeline("fill-mask", model = model, tokenizer = tokenizer, device=0 if torch.cuda.is_available() else -1)
    for occupation in tqdm(occupations):
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
            main_score = math.log2(male_score/female_score)
            occupation_scores[occupation].append(main_score)
    for occupation in occupation_scores:
        temp = occupation_scores[occupation]
        avg = sum(temp)/len(temp)
        average_scores[occupation].append(avg)
    return average_scores
def get_malor_hisher(model,tokenizer,sentences,occupations):
    avg_log_score = get_occupation_average_hisher(model,tokenizer,sentences,occupations)
    values = list(avg_log_score.values())
    return (np.mean(np.abs(values)))

#malenames-femalenames
def get_occupation_average_malefemale(model,tokenizer,sentences,occupations):
    occupation_scores = {occupation: [] for occupation in occupations}
    average_scores = {occupation: [] for occupation in occupations}
    male_names = ['Michael', 'David', 'James', 'John', 'Robert', 'William', 'Richard', 'Thomas', 'Christopher', 'Joseph', 'Steven', 'Paul', 'Daniel', 'Andrew', 'Kenneth', 'George', 'Charles', 'Stephen', 'Anthony', 'Edward', 'Brian', 'Ronald', 'Kevin', 'Matthew', 'Jason', 'Timothy', 'Gary', 'Jeffrey', 'Scott']
    female_names = ['Jennifer', 'Linda', 'Patricia', 'Susan', 'Mary', 'Sarah', 'Jessica', 'Elizabeth', 'Karen', 'Nancy', 'Lisa', 'Margaret', 'Betty', 'Sandra', 'Ashley', 'Dorothy', 'Kimberly', 'Emily', 'Michelle', 'Laura', 'Rebecca', 'Amanda', 'Carol', 'Helen', 'Sharon', 'Cynthia', 'Kathleen', 'Amy', 'Melissa']
    male_names = [name.lower() for name in male_names]
    female_names = [name.lower() for name in female_names]
    all_names = male_names + female_names
    nlp = pipeline("fill-mask", model = model, tokenizer = tokenizer, device=0 if torch.cuda.is_available() else -1)
    for occupation in tqdm(occupations):
        for sentence in sentences:
            sentence_structure = sentence.replace("[MASK]",f"{nlp.tokenizer.mask_token}")
            sentence_structure = sentence_structure.replace("[OCC]",occupation)
            output = nlp(sentence_structure, targets=all_names, top_k=58)
            male_prob = [item['score'] for item in output if item['token_str'] in male_names]
            female_prob = [item['score'] for item in output if item['token_str'] in female_names]
            male_prob_avg = sum(male_prob) / len(male_prob)
            female_prob_avg = sum(female_prob) / len(female_prob)
            main_score = math.log2(male_prob_avg/female_prob_avg)
            occupation_scores[occupation].append(main_score)
    for occupation in occupation_scores:
        temp = occupation_scores[occupation]
        avg = sum(temp)/len(temp)
        average_scores[occupation].append(avg)
    return average_scores
def get_malor_malefemale(model,tokenizer,sentences,occupations):
    avg_log_score = get_occupation_average_malefemale(model,tokenizer,sentences,occupations)
    values = list(avg_log_score.values())
    return (np.mean(np.abs(values)))


