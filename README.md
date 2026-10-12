
# Exploration and Mitigation of Gender Bias in Word Embeddings from Transformer-based Language Models

Machine learning can uncover data biases resulting from human error when implemented without proper restraint. However, this complexity often arises from word embedding, a prominent technique for capturing textual input as vectors applied in different machine learning and natural language processing tasks. Word embeddings are biased because they are trained on text data, which frequently incorporates prejudice and bias from society. These biases may become deeply established in the embeddings, producing unfair or biased results in AI applications. Efforts are made to recognize and lessen certain prejudices, but comprehensive bias elimination is still tricky. In Natural Language Processing (NLP) systems, contextualized word embeddings are now more commonly used for representing texts than traditional embeddings. It is also critical to evaluate biases contained in these since biases of various kinds have already been discovered in standard word embeddings. We focus on transformer-based language models including BERT, ALBERT, RoBERTa and DistilBERT which produce contextual word embeddings. To measure the extent to which gender biases exist, we calculate the probabilities of filling MASK by the models and based on this, we develop a novel metric, MALoR, to measure gender bias. We then continue pretraining these models on a gender-balanced dataset to mitigate the bias. Gender balanced dataset is created by applying Counterfactual Data Augmentation (CDA). To ensure consistency, we perform our experiments on different gender pronouns and nouns - “he-she,” “his-her,” and “male names-female names.”  Finally, using a downstream task, we verify that our debiased models show no performance drop. These debiased models can then be used across several applications.



## Environment Setup

First, create an environment with python=3.8. Set the "name" as your prefered environment name.

```bash
conda create -n name python=3.8
conda activate name
```

Next, install all the required packages

```bash
pip install numpy pandas nltk transformers tensorflow scipy matplotlib seaborn
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```
    
## Running Script

To run the script, clone the repository and go to the folder named "Code".

```bash
git clone https://github.com/AriyanH100/bias-in-word-embeddings.git
cd Code
```


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


## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@misc{hossain2025exploringmitigatinggenderbias,
  title={Exploring and Mitigating Gender Bias in Encoder-Based Transformer Models},
  author={Hossain, Ariyan and Hannan, Khondokar Mohammad Ahanaf and Haque, Rakinul and Rafa, Nowreen Tarannum and Musarrat, Humayra and Dipu, Shoaib Ahmed and Sadeque, Farig Yousuf},
  year={2025},
  eprint={2511.00519},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.00519}
}
```

Paper: [Exploring and Mitigating Gender Bias in Encoder-Based Transformer Models](https://arxiv.org/abs/2511.00519)
