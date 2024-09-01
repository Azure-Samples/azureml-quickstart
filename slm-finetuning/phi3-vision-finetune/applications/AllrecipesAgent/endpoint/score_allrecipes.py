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
        img_data = image_file[len(img_base64_pref):]
        print("Image data:", img_data)
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

#{"input_data": {"input_string": ["what do you see?"]}, "parameters": {"top_p": 1.0, "temperature": 1.0, "max_new_tokens": 500}}

def process_actions_string(
        input_string = "You are a useful AI that searches AllRecipes.com website for various recipies.  The following json document contains a set of keyboard and mouth actions together with the screenshots that preempted them to search for 'italian wedding soup' recipe on the website.  Suggest the nest set of keyboard and mouth actions to continue searching for the recipe. ['<sleep>11.645689', '<image>screenshot_2024-08-31_11-24-34.961750', '<mouse>on_click(1070,182,Button.left,True)', '<sleep>3.881456', '<image>screenshot_2024-08-31_11-24-40.230237', 'italian', '<key>Key.space:True', 'wedding', '<key>Key.space:True']	['soup', '<sleep>1.752508']",
        image_loader = None,
        action_updater = lambda image, img_cnt, actions_image_url: f"<|image_{img_cnt}|>"
):

    input_string = input_string.split("\t")[0]
    prompt_string = input_string.split("[")[0]
    
    print("----------------------------------")
    print(f"INPUT_STRING: {input_string[:500]}")
    print("----------------------------------")

    delim_idx = input_string.index("[")

    actions_json = input_string[delim_idx:].replace("'", "\"")
    actions_json = json.loads(actions_json)
    #print(actions_json)


    images = []
    img_cnt = 0
    for i in range(len(actions_json)):
        action = actions_json[i]

        if action.startswith("<image>"):
            img_cnt += 1
            actions_image_url = action[action.index('>')+1: ]
            
            image = image_loader(actions_image_url)

            images.append(image)

            actions_json[i] = "".join(["<image>", action_updater(image, img_cnt, actions_image_url)])

    actions_string = json.dumps(actions_json)

    return f"<|user|>\n{prompt_string} {actions_string}<|end|><|assistant|>\n", images

def run(raw_data = {
        "prompt" : "<|user|>\n<|image_1|>What is shown in this image?<|end|><|assistant|>\n",
        "image_url" : "https://th.bing.com/th/id/OIP.dep14_-r-TaqPFIrmI4HBAHaHa?rs=1&pid=ImgDetMain"
    }):

    print("===============================")
    print(raw_data[:100])
    print("===============================")

    raw_data = json.loads(raw_data)

    if "input_data" in raw_data:
        data_str = raw_data["input_data"]["input_string"][0]
        #print("--data_str: ", data_str)
        
    
    image_prompt, images = process_actions_string(input_string = data_str, image_loader = load_image)
    
    inputs = processor(image_prompt, images, return_tensors="pt").to(device)
    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 


    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # Remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    print("response_text:", response_text)
    

    output = [{"0" : response_text}]
    print("Output:", output)

    return output
    # return {
    #     "predicted_text": response_text,
    #     "image_data_url": data_url
    # }


# if __name__ == "__main__":
#     init()
#     run()
    
    #Model Dir: /mnt/azureml/cr/j/2bd79af779d645b7addc08785fd5204a/cap/data-capability/wd/INPUT_model_dir