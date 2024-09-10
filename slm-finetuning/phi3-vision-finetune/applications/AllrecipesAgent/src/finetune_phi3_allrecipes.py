# Import necessary libraries
# Code orginally from https://wandb.ai/byyoung3/mlnews3/reports/How-to-fine-tune-Phi-3-vision-on-a-custom-dataset--Vmlldzo4MTEzMTg3 
# Credits to: Brett Young https://github.com/bdytx5/

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoProcessor
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
import json


import torch.optim as optim
import torch.nn.functional as F
import random as rand

torch.manual_seed(3)


from jinja2 import Environment, FileSystemLoader
env = Environment(loader = FileSystemLoader('templates'))
#template = env.get_template('system_prompt_template.v2.jinja')
template = env.get_template(sys.argv[3])

JSoA = open("./json/user_actions_definitions.schema.json").read()

def render_template(recipe, JSoA, JAoA):
    output = template.render(website = "AllRecipes.com",
                            recipe=recipe,
                            JSoA = JSoA,
                            JAoA = JAoA
                            )
    return output

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

        row = self.dataframe.iloc[idx]

        target_name = row[0]
        training_data = row[1]

        training_prompt_data = json.loads(training_data)

        pixel_values_array = []

        img_cnt = 0
        for i in range(len(training_prompt_data)):
            data = training_prompt_data[i]

            if data["type"] == "image":
                img_cnt += 1
                
                image_file = data["path"]
                training_prompt_data[i]["path"] = f"<|image_{img_cnt}|>"
                
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

        random_split = rand.randint(1, len(training_prompt_data)-1)
        
        training_input = json.dumps(training_prompt_data[:-random_split])
        training_output = json.dumps(training_prompt_data[random_split:])

        training_prompt = render_template(target_name, JSoA=JSoA, JAoA=training_input)

        text = f"<|user|>{training_prompt}<|end|><|assistant|>{training_output}<|end|>"

        # Tokenize the text input
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
        encodings['pixel_values'] = pixel_values_array

        return {key: torch.tensor(val) for key, val in encodings.items()}

    def image_transform_function(self, image):
        # Convert the image to a numpy array
        image = np.array(image)
        return image


# Initialize processor and tokenizer for the pre-trained model
#model_id = "microsoft/Phi-3-vision-128k-instruct"
model_id = "microsoft/Phi-3.5-vision-instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = processor.tokenizer


# Load dataset from disk
data_dir = sys.argv[1]

#dataset_path = f"{data_dir}/keylog_aug.txt"
dataset_path = f"{data_dir}/{sys.argv[4]}"
print("Dataset:", dataset_path)
output_path = sys.argv[2]
df = pd.read_csv(dataset_path, sep="\t", header=None, index_col=False)
#df = pd.read_csv(dataset_path)


#if False:

# Split dataset into training and validation sets
train_size = int(0.9 * len(df))
val_size = len(df) - train_size
train_indices, val_indices = random_split(range(len(df)), [train_size, val_size])
train_indices = train_indices.indices
val_indices = val_indices.indices
train_df = df.iloc[train_indices]
val_df = df.iloc[val_indices]

#train_df = df
#val_df = df

print("Train_df", train_df)

row = train_df.iloc[0]
print("DF Row: ", row)

print("val Row: ", row[0])




# Create dataset and dataloader for training set
#train_dataset = BurberryProductDataset(train_df, tokenizer, max_length=512, image_size=128, image_dir=data_dir)
#train_dataset = SujetProductDataset(train_df, tokenizer, max_length=512, image_size=128, image_dir=data_dir)
train_dataset = AllrecipesCaptureDataset(train_df, tokenizer, max_length=512, image_size=128, image_dir=data_dir)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Create dataset and dataloader for validation set
#val_dataset = BurberryProductDataset(val_df, tokenizer, max_length=512, image_size=128, image_dir=data_dir)
#val_dataset = SujetProductDataset(val_df, tokenizer, max_length=512, image_size=128, image_dir=data_dir)
val_dataset = AllrecipesCaptureDataset(val_df, tokenizer, max_length=512, image_size=128, image_dir=data_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map="cuda", 
                                             trust_remote_code=True, 
                                             torch_dtype="auto",
                                             _attn_implementation='flash_attention_2' )

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 5
#eval_interval = 150  # Evaluate every 'eval_interval' steps
eval_interval = 50
loss_scaling_factor = 1000.0  # Variable to scale the loss by a certain amount
save_dir = f'./saved_models'
step = 0
accumulation_steps = 64  # Accumulate gradients over this many steps

# Create a directory to save the best model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

best_val_loss = float('inf')
best_model_path = None

# Select 10 random images from the validation set for logging
num_log_samples = 2
log_indices = random.sample(range(len(val_dataset)), num_log_samples)

# Function to extract the predicted price from model predictions
def extract_price_from_predictions(predictions, tokenizer):
    # Assuming the price is at the end of the text and separated by a space
    predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
    try:
        predicted_price = float(predicted_text.split()[-1].replace(',', ''))
    except ValueError:
        predicted_price = 0.0
    return predicted_price

# Function to evaluate the model on the validation set
def evaluate(model, val_loader, device, tokenizer, step, log_indices, max_samples=None):
    model.eval()
    total_loss = 0
    total_price_error = 0
    log_images = []
    log_gt_texts = []
    log_pred_texts = []
    #table = wandb.Table(columns=["Image", "Ground Truth Text", "Predicted Text"])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples and i >= max_samples:
                break

            if batch is None:  # Skip if the batch is None
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()
            #actual_price = batch['price'].item()

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Calculate price error
            # predictions = torch.argmax(outputs.logits, dim=-1)
            # predicted_price = extract_price_from_predictions(predictions, tokenizer)
            # price_error = abs(predicted_price - actual_price)
            # total_price_error += price_error

            # Log images, ground truth texts, and predicted texts
            # if i in log_indices:
            #     log_images.append(pixel_values.cpu().squeeze().numpy())
            #     log_gt_texts.append(tokenizer.decode(labels[0], skip_special_tokens=True))
            #     log_pred_texts.append(tokenizer.decode(predictions[0], skip_special_tokens=True))

            #     # Convert image to PIL format
            #     pil_img = to_pil_image(resize(torch.from_numpy(log_images[-1]).permute(2, 0, 1), (336, 336))).convert("RGB")
                
            #     # Add data to the table
            #     table.add_data(wandb.Image(pil_img), log_gt_texts[-1], log_pred_texts[-1])

                # Log the table incrementally
    # wandb.log({"Evaluation Results step {}".format(step): table, "Step": step})

    avg_loss = total_loss / (i + 1)  # i+1 to account for the loop index
    # avg_price_error = total_price_error / (i + 1)
    model.train()

    return avg_loss #, avg_price_error

# Set the model to training mode
model.train()

# Training loop for the specified number of epochs
for epoch in range(num_epochs):
    total_train_loss = 0
    total_train_price_error = 0
    batch_count = 0

    for batch in train_loader:
        step += 1

        if batch is None:  # Skip if the batch is None
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = input_ids.clone().detach()
        #actual_price = batch['price'].float().to(device)

        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            labels=labels
        )
        loss = outputs.loss
        total_loss = loss
        predictions = torch.argmax(outputs.logits, dim=-1)            
        #predicted_price = extract_price_from_predictions(predictions, tokenizer)

        total_loss.backward()

        if (step % accumulation_steps) == 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= accumulation_steps
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += total_loss.item()
        #total_train_price_error += abs(predicted_price - actual_price.item())
        batch_count += 1

        # Log batch loss to Weights & Biases
        print({"Batch Loss": total_loss.item(), "Step": step})

        print(f"Epoch: {epoch}, Step: {step}, Batch Loss: {total_loss.item()}")

        if step % eval_interval == 0:
            #val_loss, val_price_error = evaluate(model, val_loader, device, tokenizer=tokenizer, log_indices=log_indices, step=step )
            val_loss = evaluate(model, val_loader, device, tokenizer=tokenizer, log_indices=log_indices, step=step )
            print({
                "Validation Loss": val_loss,
#                "Validation Price Error (Average)": val_price_error,
                "Step": step
            })
#            print(f"Step: {step}, Validation Loss: {val_loss}, Validation Price Error (Normalized): {val_price_error}")
            print(f"Step: {step}, Validation Loss: {val_loss}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_path #os.path.join(save_dir, f"best_model")
                model.save_pretrained(best_model_path, safe_serialization=False)
                tokenizer.save_pretrained(best_model_path)

            avg_train_loss = total_train_loss / batch_count
            #avg_train_price_error = total_train_price_error / batch_count
            print({
                "Epoch": epoch,
                "Average Training Loss": avg_train_loss,
                #"Average Training Price Error": avg_train_price_error
            })
            
    #print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}, Average Training Price Error: {avg_train_price_error}")
    #print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")


    print("Best Model Path: ", best_model_path)
    # Log the best model to Weights & Biases
    # if best_model_path:
    #     run.log_model(
    #         path=best_model_path,
    #         name="phi3-v-burberry",
    #         aliases=["best"],
        # )

# Finish the Weights & Biases run
#wandb.finish()