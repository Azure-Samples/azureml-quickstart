import os
import logging
import json
import numpy
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import base64
import io

def load_image(image_file):
    img_base64_pref = 'data:image/jpeg;base64,'

    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif image_file.startswith(img_base64_pref):
        img_data = image_file[len(img_base64_pref):]
        print("Image data:", img_data)
        msg = base64.b64decode(img_data)
        buf = io.BytesIO(msg)
        image = Image.open(buf).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    
    return image

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model,model_name, image_processor, tokenizer, context_len, roles, args

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    

    model_path = os.getenv("AZUREML_MODEL_DIR")
    print("Model path: ", model_path)
    print(os.listdir())

    model_path = os.path.join(model_path, "llava_out_dir")

    disable_torch_init()

    class LocalArgs:
        def __init__(self) -> None:
            self.model_base = None
            self.load_8bit = False
            self.load_4bit = True
            self.device = "cuda"
            self.temperature = 0.2
            self.conv_mode = None
            self.max_new_tokens = 512
            self.debug = True

    args = LocalArgs()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    args.conv_mode = conv_mode

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    

    # deserialize the model file back into a sklearn model
    #model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """

    print("----------------------- RAW DATA -----------------------------")
    print(raw_data)
    print("------------------------ ~~~~ --------------------------------")

    json_data = json.loads(raw_data)



    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    logging.info("model 1: request received")
    #image_file = "https://llava-vl.github.io/static/images/view.jpg" #json.loads(raw_data)["data"]
    
    image_file = json_data["image_file"]
    image_prompt = json_data["image_prompt"]

    image = load_image(image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

#    while True:
    #try:
    inp = image_prompt #"Describe the image:" #input(f"{roles[0]}: ")
    #except EOFError:
    #    inp = ""
    #if not inp:
    #    print("exit...")
    #    break

    print(f"{roles[1]}: ", end="")

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        image = None
    
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return [outputs]
    #return result.tolist()

# if __name__ == "__main__":
    
#     for i in range(5):
#         if i == 0:
#             print(f" -------------------- INIT {i} ----------------------")
#             init()
#             print(" -------------------- INIT ----------------------")
#         print(f" -------------------- RUN {i} ----------------------")
#         run(None)
#         print(" -------------------- DONE RUN ----------------------")






