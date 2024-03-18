## Bias Detection
To detect the bias present in the pretrained models, run the following command. This command detects bias for "bert-base-uncased" in "he-she" experiment. Default model is "bert-base-uncased". There are two detection method - MALoR Score (detect=1) and Probability Bar Charts (detect=2).

```bash
python main.py --exp he_she --model bert-base-uncased --sent templates_he_she.csv --occ occupations.csv  --detect 3 --out he_she_bertbase_detect  
```

    --exp       select experiment = he_she / his_her / male_female, required=True
    --model      which model to use = bert-base-uncased / bert-large-uncased / distilbert-base-uncased / roberta-base / roberta-large / albert-base-v1 / albert-large-v1, required=False
    --sent      .csv file with sentence templates for bias evaluation = templates_he_she.csv / templates_his_her.csv / templates_male_female.csv, required=True
    --occ       .csv file with occupations for bias evaluation = occupations.csv, required=True
    --detect     0 (none) / 1 (malor) / 2 (probability bar) / 3 (both), required=False, default=3
    --out        output directory, required=True



## Bias Mitigation
To mitigate the bias present in the pretrained models, run the following command. This command mitigates bias for "bert-base-uncased" in "he-she" experiment. Default model is "bert-base-uncased". Debiasing is not possible in "male-female" experiment in RoBERTa and ALBERT models due to lack of the male/female names in their vocabulary.

```bash
python main.py --exp he_she --model bert-base-uncased --sent templates_he_she.csv --occ occupations.csv --dataset he_she_dataset.csv --detect 3 --out he_she_bertbase --batch 32 --epochs 200 --seed 42
```

    --exp       select experiment = he_she / his_her / male_female, required=True
    --model     which model to use = bert-base-uncased / bert-large-uncased / distilbert-base-uncased / roberta-base / roberta-large / albert-base-v1 / albert-large-v1, required=False
    --sent      .csv file with sentence templates for bias evaluation = templates_he_she.csv / templates_his_her.csv / templates_male_female.csv, required=True
    --occ       .csv file with occupations for bias evaluation = occupations.csv, required=True
    --dataset   .csv file with sentences for training = he_she_dataset.csv /his_her_dataset.csv / male_female_dataset.csv, required=False
    --detect     0 (none) / 1 (malor) / 2 (probability bar) / 3 (both), required=False, default=3
    --out        output directory, required=True
    --batch      fix batch-size for fine-tuning, required=False, default=32
    --epochs     fix epoch-size for fine-tuning, required=False, default=200
    --seed       fix seed-size required=False, default=42
