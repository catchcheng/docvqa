import ast
from collections import Counter
from typing import Tuple

# todo: remove unused imports
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch.nn.functional as F
import os
import shutil
from PyPDF2 import PdfReader, PdfWriter
from keras import backend as K
import torch

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import tqdm
from tqdm import tqdm
tqdm.pandas()



class DocVQA_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        question = sample["question"]
        ocr_text = sample["ocr_text"]
        answer = sample["answers"]  # Can be a list or a single string
        # import time
        # print(question, ocr_text, answer)
        # time.sleep(20)
    
        # Ensure answers are properly formatted
        def extract_first_answer(answer):
            try:
                parsed_answers = ast.literal_eval(answer)  # Convert string list to Python list
                if isinstance(parsed_answers, list) and len(parsed_answers) > 0:
                    return parsed_answers[0]  # Take the first answer
            except (SyntaxError, ValueError):
                pass
            return ""  # Return empty string if there's an issue
        answer = extract_first_answer(answer)

        prompt = f"question: {question} context: {ocr_text}"
        # Tokenize input (question + OCR text as context)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Process answer (convert to label if needed)
        labels = self.tokenizer(
            answer, 
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )["input_ids"]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
        
        

# todo: retrain model with new labels

# This classifier take the text information from the previous page and 
# current page as input and predict if the current page is a document's first page
class T5_doc_vqa:
    def __init__(self, force_train: bool = False):

        # CONSTANTS
        self.MAX_TOKENIZER_LENGTH = 512
        self.LEARNING_RATE = 3e-4
        self.MAX_EPOCHS = 5
        self.BATCH_SIZE = 4
        self.MODEL_FILE_NAME = "t5"
        # END CONSTANTS

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        model_path = os.path.join(r"C:\Users\15392\github_repos\docvqa", self.MODEL_FILE_NAME)
        if force_train or not os.path.exists(model_path):
            self.train()
        else:
            # self.model = BertForSequenceClassification.from_pretrained(
            #     "bert-base-multilingual-cased", num_labels=2
            # )
            self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        # define the one-hot encoder, all the labels will be convert into one-hot version during the training and prediction
        
    def generate_answers(self, dataset):
        inputs = self.tokenizer(dataset["input_text"], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=128)
        
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions
    
    def load_docvqa_data(self, file_path) -> DocVQA_Dataset:
        
        # Read CSV with error handling
        df = pd.read_csv(file_path)
        
        # Reset index to ensure continuous indexing
        df.reset_index(drop=True, inplace=True)
        
        
        return DocVQA_Dataset(df, self.tokenizer, max_length=self.MAX_TOKENIZER_LENGTH)
    
        
    # Tokenize dataset
    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
        labels = self.tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def train(self):
        # Load dataset
        train_dataset = self.load_docvqa_data("data/dataset/train_dataset_with_ocr.csv")
        valid_dataset = self.load_docvqa_data("data/dataset/val_dataset_with_ocr.csv")
        
        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # Load pre-trained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device : ", device)
        self.model.to(device)

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./t5-docvqa",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            gradient_accumulation_steps=self.BATCH_SIZE,  # Adjust for GPU memory
            num_train_epochs=self.MAX_EPOCHS,
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.tokenizer
        )

        # Train the model
        trainer.train()

        