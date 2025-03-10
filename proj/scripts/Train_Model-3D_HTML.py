import json
import os
import sys
from tqdm.auto import tqdm
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig


sys.path.insert(0, '../src')
from modeling_tabeleiro import TabeleiroModel
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


with open("msg_3D_HTML.json", 'w') as out:
        json.dump({'outputs': []}, out, ensure_ascii=False, indent=4)

def write_msg(msg):
    with open("msg_3D_HTML.json", encoding="utf-8") as f:
        json_data = json.load(f)
    
    with open("msg_3D_HTML.json", 'w') as out:
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
        self.transform = transforms.Compose([
            v2.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 0.2)),
            v2.ColorJitter(brightness=.1, hue=.2),
            v2.RandomRotation(degrees=1, expand=True, interpolation=Image.BILINEAR, fill=(255,255,255)),
            v2.JPEG((70, 100)),
            v2.RandomPerspective(distortion_scale=0.03, p=0.3, interpolation=Image.BILINEAR, fill=(255,255,255))
        ])
        
    def __len__(self):
        return len(self.annotations)
    
    
    def __getitem__(self, idx):
        
        file_name = self.annotations[idx]
        
        with open(ANN_PATH + file_name + "-HTML.json", encoding="utf-8") as f:
            annotation = json.load(f)
        
        image = self.transform(Image.open(IMAGE_PATH + file_name + IMG_FORMAT).convert("RGB"))
        
        
        # inputs
        pixel_values = processor(image, random_padding=self.split == "train", return_tensors="pt").pixel_values.squeeze()

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


cell_tokens = [processor.tokenizer.convert_tokens_to_ids([cell])[0] for cell in ['<td>', '<td']]
row_tokens = [processor.tokenizer.convert_tokens_to_ids([row_type])[0] for row_type in ['<tr>']]


model = TabeleiroModel.from_pretrained(MODELS_PATH+"donut-base",
                                       from_donut=True,
                                       decoder_extra_config={"pos_counters":[cell_tokens, row_tokens]},
                                       donut_config = config)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<table_extraction>"])[0]


#CREATE DATASET AND DATALOADER
train_dataset = DonutTableDataset(json_list,
                             max_length = max_length,
                             image_size = image_size)

train_dataloader = DataLoader(train_dataset, batch_size=20, num_workers=8, shuffle=True)



# TRAIN MODEL
avg_size = 500 #moving avg size

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = torch.nn.DataParallel(model, device_ids=range(4))
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
        if num_steps%12000 == 0 :
            model.module.save_pretrained("../../aux/models/by_step/3D_HTML/model_3D_HTML-STEP_"+str(num_steps))
            
        if scheduler.get_last_lr()[0] > 1e-5:
            scheduler.step() 
                
        if i % avg_size == 0:
            print(str(i) + " Loss: ", mean_smpl_loss/avg_size)
            write_msg("batch " + str(i) +" loss: "+ str(mean_smpl_loss/avg_size))
            mean_smpl_loss = 0 
       
        #print(os.system("nvidia-smi"))
        
    
        
    model.module.save_pretrained("../../aux/models/checkpoints/model_3D_HTML-checkpoint-epoch_"+str(epoch))
    print("Epoch's mean loss: ", mean_loss/len(train_dataloader))
    
    write_msg("Epoch checkpointed: " + str(epoch+1) +" \n"+
              "Epoch's mean Loss: " + str(mean_loss/len(train_dataloader)))
 
              
model.module.save_pretrained("../../aux/models/model-3D_HTML-3_EPOCHS")
