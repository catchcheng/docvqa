from openai import OpenAI
import time
from typing import List, Dict
import os
import base64
import pandas as pd
from tqdm import tqdm


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
def gptqa(prompt: str, image_path: str, openai_model_name: str, system_message: str):
    
    client = OpenAI()
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
    return completion.choices[0].message.content, completion.usage


def tload(f, mode="r"):
    with open(f, mode) as file:
        output = file.read()
    return output


def extract_structured_ocr(row):
    image_path = "data/spdocvqa_images/" + row['image'].split('/')[-1]
    # print(ocr_path)
    verification_prompt = tload('prompt_template.txt')
    try:
        user_prompt = row['ocr_text']
        completion, _ = gptqa(user_prompt, image_path, "gpt-4o-mini", verification_prompt)
    except Exception as e:
        completion = ""
        print(f"Error processing {row['image'].split('/')[-1]}: {e}")
    return completion


import concurrent.futures
def get_structured_ocr():
    df = pd.read_csv("dataset_with_ocr.csv")[:10]

    # Run OCR extraction in parallel with tqdm for progress tracking
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(extract_structured_ocr, [row for _, row in df.iterrows()]), total=len(df)))
    df['structured_ocr_text'] = results
    df.to_csv("dataset_add_structured_ocr.csv", index=False)


if __name__ == "__main__":
    get_structured_ocr()
