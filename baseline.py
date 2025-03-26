import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DonutProcessor, VisionEncoderDecoderModel
import torch
from tqdm import tqdm
from PIL import Image

def create_datase_with_concat_ocr(name):
    with open(f'Annotations/{name}_v1.0_withQT.json') as f:
        data = json.load(f)



    pd.set_option('display.max_colwidth', None)

    df = pd.DataFrame(data['data'])

    root_dir = "data/"
    root_dir_ocr = root_dir + "spdocvqa_ocr/"
    def load_ocr_json(root_dir_ocr):
        with open(root_dir_ocr, 'r') as f:
            return json.load(f)

    def extract_text_from_ocr(ocr_json):
        recognition_results = ocr_json.get('recognitionResults', [])
        all_lines = []
        for result in recognition_results:
            lines = result.get('lines', [])
            for line in lines:
                text_line = line.get('text', '')
                all_lines.append(text_line)
        # Join all lines into one string, with newlines between them.
        return "\n".join(all_lines)

    def extract_text_from_ocr_sorted(ocr_json):
        recognition_results = ocr_json.get('recognitionResults', [])
        all_lines = []
        for result in recognition_results:
            lines = result.get('lines', [])
            # Sort by the y-coordinate (second value in boundingBox)
            # sorted_lines = sorted(lines, key=lambda line: line.get('boundingBox', [0, 0])[1])
            # for line in sorted_lines:
            for line in lines:
                text_line = line.get('text', '')
                all_lines.append(text_line)
        return "\n".join(all_lines)

    # And you have a function that maps the image path to its OCR JSON file path
    def extract_and_add_ocr_text(row):
        ocr_path = root_dir_ocr + row['ucsf_document_id'] + "_" + row['ucsf_document_page_no'] + ".json"
        # print(ocr_path)
        try:
            ocr_json = load_ocr_json(ocr_path)
            # Use sorted extraction if needed; otherwise, use extract_text_from_ocr
            ocr_text = extract_text_from_ocr(ocr_json)
        except Exception as e:
            ocr_text = ""
            print(f"Error processing {ocr_path}: {e}")
        return ocr_text

    # Create a new column with OCR text
    df['ocr_text'] = df.apply(extract_and_add_ocr_text, axis=1)
    df.to_csv(f"data/dataset/{name}_dataset_with_ocr.csv", index=False)

def run_T5_xl():
    df = pd.read_csv("dataset_with_ocr.csv")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-Large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-Large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    def get_answer(ocr_text, question, max_length=50):
        # Create the prompt
        prompt = f"Document: {ocr_text}\nQuestion: {question}\nAnswer:"
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move all inputs to the same device as the model
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate the answer
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return only the answer part, not including the prompt
        answer = answer.replace(prompt, "").strip()
        return answer
    
    
    # Create a list to store answers with progress tracking
    answers = []
    # Use tqdm to wrap the iteration
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Questions"):
        answer = get_answer(row['ocr_text'], row['question'])
        answers.append(answer)
    
    # Add answers to the dataframe
    df['T5_generated_answer'] = answers
    
    df.to_csv("dataset_with_ocr_T5_solution.csv")
    
def run_donut():
    # Load dataset
    df = pd.read_csv("dataset_with_ocr.csv")
    df = df.head(10)
    # Load Donut model and processor
    model_name = "naver-clova-ix/donut-base"  # You can change this to "donut-large" if needed
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    def get_answer(image_path, question, max_length=50):
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Format the question prompt
        prompt = f"<s_question>{question}</s_question>"
        
        # Tokenize the prompt
        inputs = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate the answer
        outputs = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and return the answer
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return answer.strip()
    
    # Process each row
    answers = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        answer = get_answer("data/spdocvqa_images/" + row['image'].split('/')[-1], row['question'])
        answers.append(answer)
    
    # Add answers to the dataframe
    df['Donut_generated_answer'] = answers
    
    # Save the results
    df.to_csv("dataset_with_donut_solution.csv", index=False)

import Levenshtein
def count_similarity():
    
    
    df = pd.read_csv("dataset_with_donut_solution.csv")
    df.reset_index(drop=True, inplace=True)
    import Levenshtein
    import re

    # Function to clean and preprocess text
    def clean_text(text):
        # Remove extra spaces and punctuation (optional depending on your needs)
        text = re.sub(r'[^\w\s]', '', str(text))  # Remove punctuation
        text = text.lower().strip()  # Convert to lowercase and strip spaces
        return text

    # Normalized Levenshtein similarity
    def normalized_levenshtein_similarity(s1, s2):
        s1, s2 = clean_text(s1), clean_text(s2)
        dist = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - (dist / max_len)

    # Function to compute the similarity for a single row
    def row_similarity(row):
        # Ensure 'answers' is a list (if it's a single string, wrap it in a list)
        ground_truths = row['answers'] if isinstance(row['answers'], list) else [row['answers']]
        
        # Get the generated answer
        generated = row['Donut_generated_answer']
        
        # Compute similarity with each ground truth answer
        similarities = [normalized_levenshtein_similarity(gt, generated) for gt in ground_truths]
        
        # Return the maximum similarity (best match) as the score for that row.
        return max(similarities) if similarities else 0.0

    # Compute similarity for each row
    df["similarity"] = df.apply(row_similarity, axis=1)

    # Calculate the average similarity (ANSL)
    ansl_accuracy = df["similarity"].mean()

    print("Row similarities:")
    print(df[["answers", "Donut_generated_answer", "similarity"]])
    print("Average Normalized Levenshtein Similarity (ANSL):", ansl_accuracy)


def docling():
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert('data/spdocvqa_images/ffbf0023_4.png')
    
    print(result.document.export_to_markdown())













def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]
    return output



import argparse
import torch
from PIL import Image
from donut import DonutModel

# def process_single_image(image_path, task_name="cord-v2", pretrained_path="naver-clova-ix/donut-base-finetuned-cord-v2"):
#     """
#     Process a single image using Donut model and return JSON output
    
#     Args:
#         image_path (str): Path to the input image
#         task_name (str, optional): Task type. Defaults to "cord-v2"
#         pretrained_path (str, optional): Pretrained model path. Defaults to Donut CORD-v2 model
    
#     Returns:
#         dict: Model inference predictions
#     """
#     # Create task-specific prompt
#     if task_name == "docvqa":
#         raise ValueError("This script is optimized for non-VQA tasks. Use demo_process_vqa for DocVQA.")
    
#     task_prompt = f"<s_{task_name}>"
    
#     pretrained_model = DonutModel.from_pretrained(pretrained_path,ignore_mismatched_sizes=True)

#     if torch.cuda.is_available():
#         pretrained_model.half()
#         device = torch.device("cuda")
#         print(device)
#         pretrained_model.to(device)
#     else:
#         pretrained_model.encoder.to(torch.bfloat16)

#     pretrained_model.eval()
    
#     # Open and process the image
#     input_img = Image.open(image_path)
    
    
#     # Run inference
#     output = pretrained_model.inference(image=input_img, prompt=task_prompt)["predictions"][0]
    
#     return output
import re

from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch


def main():
    
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # load document image
    image = Image.open("data/spdocvqa_images/ffbf0023_4.png").convert("RGB")

    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))
    
    # # Process the image
    # result = process_single_image(
    #     image_path="data/spdocvqa_images/ffbf0023_4.png"
    # )
    
    # # Print the result
    # import json
    # print(json.dumps(result, indent=2))

if __name__ == "__main__":
    create_datase_with_concat_ocr("val")
