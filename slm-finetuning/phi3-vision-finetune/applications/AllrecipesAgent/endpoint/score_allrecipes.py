from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
from transformers import AutoTokenizer
import os
import base64
from io import BytesIO
import torch
from pathlib import Path
import json


def load_image(image_file):
    img_base64_pref = 'data:image/jpeg;base64,'

    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif image_file.startswith(img_base64_pref):
        print("base64 image detected")
        img_data = image_file[len(img_base64_pref):]
        print("Loaded image data")
        #print("Image data:", img_data)
        msg = base64.b64decode(img_data)
        buf = BytesIO(msg)
        image = Image.open(buf).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    
    return image

EXT_TO_MIMETYPE = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.svg': 'image/svg+xml'
}


def image_to_data_url(image: Image.Image, ext: str) -> str:
    ext = ext.lower()
    if ext not in EXT_TO_MIMETYPE:
        ext = '.jpg'  # Default to .jpg if extension is not recognized
    mimetype = EXT_TO_MIMETYPE[ext]
    buffered = BytesIO()
    image_format = 'JPEG' if ext in ['.jpg', '.jpeg'] else ext.replace('.', '').upper()
    image.save(buffered, format=image_format)
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_url = f"data:{mimetype};base64,{encoded_string}"
    return data_url

device = 'cuda'

def init():
    global model, processor

    model_root = os.getenv("AZUREML_MODEL_DIR") 

    model_id = f"{model_root}/out_dir"
    #model_id = model_root
    base_model_id = "microsoft/Phi-3.5-vision-instruct" 
    #model_id = base_model_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True
    )

    model.to(device)

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True) 


def run(raw_data):

    print("===============================")
    #print(f"{raw_data}"[:100])
    print("===============================")

    raw_data = json.loads(raw_data)

    input_string = raw_data["input_data"]["input_string"][0]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Loaded input_string:", input_string[:1000])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    data_js = json.loads(input_string)
    print("Loaded data_js:")
    
    data_prompt = data_js["prompt"]
    data_images = data_js["images"]

    print("Prompt:", data_prompt)
    print("Images Len: ", len(data_images))

    images = []

    for data_image in data_images:
        image = load_image(data_image)
        images.append(image)

    prompt = f"<|user|>\n{data_prompt}<|end|><|assistant|>\n"
    
    inputs = processor(prompt, images, return_tensors="pt").to(device)
    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # # Remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    print("response_text:", response_text)
    

    output = [{"0" : response_text}]
    #print("Output:", output)
    print("Output received")

    return output
    # return [{"0" : test_response}]
    # return {
    #     "predicted_text": response_text,
    #     "image_data_url": data_url
    # }


# if __name__ == "__main__":
#     init()
#     run()
    
    #Model Dir: /mnt/azureml/cr/j/2bd79af779d645b7addc08785fd5204a/cap/data-capability/wd/INPUT_model_dir