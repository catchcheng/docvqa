import json
from openai import OpenAI
import time
from typing import List, Dict
import os
import base64
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from LAPDoc.lapdoc.verbalizer import  SpatialFormatVerbalizer, OCRBox



root_dir = "data/"
root_dir_ocr = root_dir + "spdocvqa_ocr/"

def load_ocr_json(root_dir_ocr):
    with open(root_dir_ocr, 'r') as f:
        return json.load(f)
    
def set_openai_private_key():
    if "OPENAI_API_KEY" not in os.environ:
        with open('openai.key', 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()

set_openai_private_key()
def create_image_object_for_query(image_path: str):
        # todo fetch image type
        image_type = "image/jpeg"
        image_data = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_type};base64,{image_data}",
            },
        }


def gptqa(prompt: str, openai_model_name: str, system_message: str,  image_path=None):
    image = None
    client = OpenAI()
    if image_path is not None:
        image = create_image_object_for_query(image_path)
        completion = client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system",
                "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        image,
                    ],
                }
            ])
    else:
        completion = client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system",
                "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ])
    return completion.choices[0].message.content, completion.usage


def tload(f, mode="r"):
    with open(f, mode) as file:
        output = file.read()
    return output


def extract_structured_ocr(row):
    image_path = "data/spdocvqa_images/" + row['image'].split('/')[-1]
    # print(ocr_path)
    verification_prompt = tload('prompt_templates/output_structured_ocr_with_image.txt')
    try:
        user_prompt = row['ocr_text']
        completion, _ = gptqa(prompt=user_prompt,  openai_model_name="gpt-4o-mini", system_message=verification_prompt, image_path=image_path)
    except Exception as e:
        completion = ""
        print(f"Error processing {row['image'].split('/')[-1]}: {e}")
    return completion

def extract_structured_ocr_text_only(row):
    # print(ocr_path)
    verification_prompt = tload('prompt_templates/output_structured_ocr_text_only.txt')
    try:
        user_prompt = row['ocr_text']
        completion, _ = gptqa(prompt=user_prompt,  openai_model_name="gpt-4o-mini", system_message=verification_prompt, image_path=None)
    except Exception as e:
        completion = ""
        print(f"Error processing {row['image'].split('/')[-1]}: {e}")
    return completion

def extract_structured_ocr_bounding_box(row):
    # print(ocr_path)
    verification_prompt = tload('prompt_templates/output_structured_ocr_with_bounding_box.txt')
    try:
        ocr_path = root_dir_ocr + str(row['ucsf_document_id']) + "_" + str(row['ucsf_document_page_no']) + ".json"
        ocr_json = load_ocr_json(ocr_path)
        recognition_results = ocr_json.get('recognitionResults', [])
        all_lines = []
        
        for result in recognition_results:
            lines = result.get('lines', [])
            page_lines = []
            for line in lines:
                text_line = line.get('text', '')
                bounding_box = line.get("boundingBox","")
                # Combine bbox string and text
                formatted_line = f"{bounding_box},{text_line}"
                page_lines.append(formatted_line)
            all_lines.append("\n".join(page_lines))
        user_prompt = "\n".join(all_lines)
        completion, _ = gptqa(prompt=user_prompt,  openai_model_name="gpt-4o-mini", system_message=verification_prompt, image_path=None)
    except Exception as e:
        completion = ""
        print(f"Error processing {row['image'].split('/')[-1]}: {e}")
    return completion
import concurrent.futures



def add_structured_ocr_from_gpt():
    df = pd.read_csv("data/dataset/val_dataset_add_spatial_ocr.csv")

    # Run OCR extraction in parallel with tqdm for progress tracking
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(extract_structured_ocr_bounding_box, [row for _, row in df.iterrows()]), total=len(df)))
    df['structured_ocr_text'] = results
    df.to_csv("data/dataset/val_dataset_add_spatial_ocr_and_bounding_box.csv", index=False)


def read_sroie_sample(page_lines: list):
    
    for l in page_lines:
        elements = l.split(',', maxsplit=8)
        x0, y0, x1, y1, x2, y2, x3, y3, text = tuple(elements)
        yield OCRBox(x0=int(x0),
                     y0=int(y0),
                     x2=int(x2),
                     y2=int(y2),
                     text=text,
                     page_index=0)


def add_spatial_ocr():
    df = pd.read_csv("data/dataset/val_dataset_with_ocr.csv")


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
    
    def extract_spatial_ocr_from_original_ocr(ocr_json):
        recognition_results = ocr_json.get('recognitionResults', [])
        all_lines = []
        
        try:
            for result in recognition_results:
                lines = result.get('lines', [])
                page_lines = []
                for line in lines:
                    text_line = line.get('text', '')
                    bounding_box = line.get("boundingBox","")
                    if bounding_box and text_line:
                        # Convert bounding box list of ints into comma-separated string
                        bbox_str = ','.join(map(str, bounding_box))
                        # Combine bbox string and text
                        formatted_line = f"{bbox_str},{text_line}"
                        page_lines.append(formatted_line)
                bboxes = list(read_sroie_sample(page_lines))
                verbalizer_instance = SpatialFormatVerbalizer()
                verbalization = verbalizer_instance(bboxes)
                all_lines.append(verbalization)
        except Exception as e:
            print(e)
            return extract_text_from_ocr(ocr_json)
        
        # Join all lines into one string, with newlines between them.
        return "\n".join(all_lines)


    # And you have a function that maps the image path to its OCR JSON file path
    def extract_and_add_ocr_text(row):
        ocr_path = root_dir_ocr + str(row['ucsf_document_id']) + "_" + str(row['ucsf_document_page_no']) + ".json"
        # print(ocr_path)
        try:
            ocr_json = load_ocr_json(ocr_path)
            # Use sorted extraction if needed; otherwise, use extract_text_from_ocr
            ocr_text = extract_spatial_ocr_from_original_ocr(ocr_json)
        except Exception as e:
            ocr_text = ""
            print(f"Error processing {ocr_path}: {e}")
        return ocr_text
        # Create a new column with OCR text
    df['spatial_ocr'] = df.apply(extract_and_add_ocr_text, axis=1)
    df.to_csv(f"data/dataset/val_dataset_add_spatial_ocr.csv", index=False)


if __name__ == "__main__":
    add_structured_ocr_from_gpt()
    # add_spatial_ocr()
