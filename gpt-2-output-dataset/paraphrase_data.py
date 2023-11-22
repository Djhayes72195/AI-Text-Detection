"""
The purpose of this module is to use a T5 paraphraser to create paraphrased versions of our data.
In doing so we hope to gain insight as to how resistant our detector is to paraphrasing attacks.

We want to be able to adjust: which data is being paraphrased, to what extent it will be paraphrased,
"""

import nltk
import json
import numpy as np
import os
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize # To split on sentence

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cpu')

ABSTRACT_PATH = 'gpt-2-output-dataset/detector/Data'
GPT_DATA_PATH = 'gpt-2-output-dataset/detector/OriginalGPTData'
PARA_PATH = "gpt-2-output-dataset/detector/ParaphrasedDataFreq={}"

print('stop')

FILE_NAMES = ['test.jsonl']

STOP_AFTER = 20 # creating short version to prototype

PARA_FREQUNCY = 3
# Add tain and validation when/if needed

def paraphrase_text(start_dir, para_dir, para_freq):
     for file_name in FILE_NAMES:
        file_path = start_dir + '/' + file_name
        para_dir = para_dir.format(para_freq)
        if not os.path.exists(para_dir):
            os.makedirs(para_dir)
        para_path = para_dir + '/' + file_name
        iterations = 0
        with open(file_path, mode='r', encoding='utf-8') as start_f, \
             open(para_path, mode='w', encoding='utf-8') as para_f:
            for row in start_f:
                iterations += 1
                record = json.loads(row)
                if record['label'] == '1':
                    para_f.write(json.dumps(record) + '\n')
                    continue # don't paraphrase human text
                if para_freq != 'whole':
                    text = sent_tokenize(record['text'])
                    for i, sentence in enumerate(text):
                        if i%para_freq == 0:
                            new_sentence = T5_paraphrase(sentence)
                            text[i] = new_sentence
                    text = " ".join(text)
                    new_record = {'text': text,
                                'label': record['label']}
                else:
                    new_text = T5_paraphrase(record['text'])
                    new_record = {'text': new_text,
                                  'label': record['label']}
                para_f.write(json.dumps(new_record) + '\n')
                if iterations > STOP_AFTER:
                    break

         

def T5_paraphrase(text):
    prompt =  "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(prompt, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu")
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=80,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )

    # Randomly select paraphrase for now.
    index_choice = np.random.choice(len(outputs))
    choice = outputs[index_choice]
    paraphrased_sent = tokenizer.decode(choice, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(paraphrased_sent)
    return paraphrased_sent

paraphrase_text(GPT_DATA_PATH, PARA_PATH, 'whole')