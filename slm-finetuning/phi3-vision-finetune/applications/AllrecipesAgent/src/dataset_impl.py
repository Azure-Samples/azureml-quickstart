import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import json

class AllrecipesCaptureDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, image_size, image_dir):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'  # Set padding side to left
        self.max_length = max_length
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the row at the given index
        row = self.dataframe.iloc[idx]
        #print("ROW::::", row)

        training_prompt = row[0]
        training_prompt_parts = training_prompt.split("[")
        training_prompt_prefix = training_prompt_parts[0]
        json_data = "["+training_prompt_parts[1]
        json_data = json_data.replace("'", "\"")
        #print(f"Json_data: {json_data}")
        training_prompt_data = json.loads(json_data)

        pixel_values_array = []

        img_cnt = 0
        for i in range(len(training_prompt_data)):
            data = training_prompt_data[i]

            if data.startswith("<image>"):
                img_cnt += 1
                training_prompt_data[i] = f"<|image_{img_cnt}|>"

                image_file = data.split(">")[1]
                
                # Get the image path from the row
                image_path = f"{self.image_dir}/images/{image_file}.png"

                try:
                    # Load and transform the image
                    image = Image.open(image_path).convert("RGB")
                    image = self.image_transform_function(image)

                    pixel_values_array.append(image)

                except (FileNotFoundError, IOError):
                    # Skip the sample if the image is not found
                    return None
                
                # Add the image and price information to the encodings dictionary
                
                #encodings['price'] = row['full_price']
        
        training_prompt_data_str = json.dumps(training_prompt_data)
        content = f"{row[1]}".strip()

        # Create the text input for the model
        #text = f"<|user|>\n<|image_1|>You are an automation agent that controls keyboard and mouse on a computer screen.  What should be the next keyboard or mouse action?<|end|><|assistant|>\{content}<|end|>"
        text = f"<|user|>{training_prompt_prefix}\n{training_prompt_data_str}<|end|><|assistant|>{content}<|end|>"

        # Tokenize the text input
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
        encodings['pixel_values'] = pixel_values_array

        return {key: torch.tensor(val) for key, val in encodings.items()}

    def image_transform_function(self, image):
        # Convert the image to a numpy array
        image = np.array(image)
        return image