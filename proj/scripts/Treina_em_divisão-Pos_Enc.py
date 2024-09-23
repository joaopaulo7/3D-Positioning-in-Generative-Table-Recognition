import json
import os
import sys
from tqdm.auto import tqdm
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig


sys.path.insert(0, '../src')
from modeling_pos_donut import PosDonutModel
from transformers import DonutProcessor


import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms



ANN_PATH = '../../aux/data/anns/train/'
IMAGE_PATH = '../../aux/data/imgs/train/'

PROCESSORS_PATH = "../../aux/processors/"
MODELS_PATH = "../../aux/models/"

IMG_FORMAT = '.png'


with open("msg_Pos.json", 'w') as out:
        json.dump({'outputs': []}, out, ensure_ascii=False, indent=4)

def write_msg(msg):
    with open("msg_Pos.json", encoding="utf-8") as f:
        json_data = json.load(f)
    
    with open("msg_Pos.json", 'w') as out:
        json_data['outputs'].append(msg)
        json.dump(json_data, out, ensure_ascii=False, indent=4)

#DEFINE DATASET CLASS
class DonutTableDataset(Dataset):
    def __init__(
        self,
        annotations,
        image_size,
        max_length,
        shuffle = True,
        split = "train",
        ignore_id = -100,
        prompt_end_token = None,
    ):            
        self.annotations = annotations
        
        
        self.image_size = image_size
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        
        
    def __len__(self):
        return len(self.annotations)
    
    
    def __getitem__(self, idx):
        
        file_name = self.annotations[idx]
        
        with open(ANN_PATH + file_name + "-HTML.json", encoding="utf-8") as f:
            annotation = json.load(f)
        
        image = Image.open(IMAGE_PATH + file_name + IMG_FORMAT)
        
        
        # inputs
        pixel_values = processor(image.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values.squeeze()
        pixel_values = pixel_values.squeeze()

        target_sequence = "<s>"+annotation+"</s>"
        
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length= max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id
        
        
        encoding = dict(pixel_values=pixel_values,
                        labels=labels,
                        target = target_sequence,
                        filename = file_name)
        
        return encoding


## LOAD FILES
json_list = os.listdir(ANN_PATH)

aux_list = []

for json_item in json_list:
    if json_item[-6] != "L":
        aux_list.append(json_item[:-5])

json_list = aux_list


#MODEL SPECS

image_size = [640, 1280]
max_length = 1500

#CONFIG AND LOAD PROCESSOR
processor = DonutProcessor.from_pretrained(PROCESSORS_PATH+"Donut_PubTables_HTML_Processor8k")
processor.image_processor.size = image_size[::-1] # should be (width, height)
processor.image_processor.do_align_long_axis = False


new_tokens  = ["<table_extraction>"]
new_tokens += ["<thead>", "</thead>", "<tbody>", "</tbody>"]
new_tokens += ["<tr>", "</tr>", "<td>", "</td>"]

new_tokens += ["<td ", ">"]
for i in range(1, 11):
    new_tokens +=['colspan="'+str(i)+'"']
    new_tokens +=['rowspan="'+str(i)+'"']
    

processor.tokenizer.add_tokens(new_tokens, special_tokens = False)


#CONFIG AND LOAD MODEL
config = VisionEncoderDecoderConfig.from_pretrained(MODELS_PATH+"donut-base")

config.encoder.image_size = image_size
config.decoder.max_position_encodings = 2048


model = PosDonutModel.from_pretrained(MODELS_PATH+"donut-base", config=config, ignore_mismatched_sizes=True)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<table_extraction>"])[0]


#CREATE DATASET AND DATALOADER
train_dataset = DonutTableDataset(json_list,
                             max_length = max_length,
                             image_size = image_size)

train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)



# TRAIN MODEL
avg_size = 400 #moving avg size

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = torch.nn.DataParallel(model, device_ids=range(1))
model.to(device) 
optimizer = torch.optim.AdamW(params=model.parameters(), lr=8e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader)//10, gamma=(0.125)**(1/20))

num_steps = 0   

for epoch in range(0, 3):
    
    print("Epoch:", epoch+1)
    mean_loss = 0
    mean_smpl_loss = 0 
    model.train()
    for i, batch in enumerate(tqdm(train_dataloader)):
            
        batch = {k: v.to(device) if k not in ["target", "filename"] else v for k, v in batch.items()}
        
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        
        loss = torch.mean(outputs.loss)
        mean_loss += loss.item()   
        mean_smpl_loss += loss.item()
        
        loss.backward()
        
        
        optimizer.step()
        optimizer.zero_grad()
        num_steps += 1
        
        if num_steps%5000 == 0 :
            model.module.save_pretrained("../../aux/models/by_step/Pos_Enc/model_Pos-STEP_"+str(num_steps))
            
        if scheduler.get_last_lr()[0] > 1e-5:
            scheduler.step() 
                
        if i % avg_size == 0:
            print(str(i) + " Loss: ", mean_smpl_loss/avg_size)
            write_msg("batch " + str(i) +" loss: "+ str(mean_smpl_loss/avg_size))
            mean_smpl_loss = 0 
       
        #print(os.system("nvidia-smi"))
        
    
        
    model.module.save_pretrained("../../aux/models/checkpoints/model_Pos-checkpoint-epoch_"+str(epoch))
    print("Epoch's mean loss: ", mean_loss/len(train_dataloader))
    
    write_msg("Epoch checkpointed: " + str(epoch+1) +" \n"+
              "Epoch's mean Loss: " + str(mean_loss/len(train_dataloader)))
 
              
model.module.save_pretrained("../../aux/models/model-Pos-3_EPOCHS")
