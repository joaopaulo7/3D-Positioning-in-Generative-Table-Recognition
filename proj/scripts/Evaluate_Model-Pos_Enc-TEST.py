import os
import sys
from tqdm.auto import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


sys.path.insert(0, '../src')
from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor
from modeling_pos_donut import PosDonutModel


IMG_PATH = "../../aux/data/imgs/val/"

#Define dataset
class DonutTableDataset(Dataset):
    def __init__(
        self,
        annotations,
        max_length,
        ignore_id = -100,
        prompt_end_token = None,
    ):            
        self.annotations_files = list(annotations.keys())
        self.annotations = annotations
        
        self.max_length = max_length
        self.ignore_id = ignore_id        
        
        
    def __len__(self):
        return len(self.annotations)
    
    
    def __getitem__(self, idx):
        
        file_name = self.annotations_files[idx]
        
        gt = self.annotations[file_name]['html']
        
        image = Image.open(IMG_PATH + file_name)
        
        
        # inputs
        pixel_values = processor(image.convert("RGB"), random_padding=False, return_tensors="pt").pixel_values.squeeze()
        pixel_values = pixel_values.squeeze()
        
        encoding = dict(file_name = file_name,
                        pixel_values=pixel_values,
                        gt = gt)
        
        return encoding

def load_model_n_processor(model_path, processor_path):
    
    #Carrega modelo
    config = VisionEncoderDecoderConfig.from_pretrained(model_path)
    model = PosDonutModel.from_pretrained(model_path, config=config) 

    #Carrega e configura o processador
    processor = DonutProcessor.from_pretrained(processor_path)
    processor.image_processor.size = model.encoder.config.image_size[::-1] # should be (width, height)
    processor.image_processor.do_align_long_axis = False
    
    new_tokens  = ["<table_extraction>"]
    new_tokens += ["<thead>", "</thead>", "<tbody>", "</tbody>"]
    new_tokens += ["<tr>", "</tr>", "<td>", "</td>"]

    new_tokens += ["<td ", ">"]
    for i in range(1, 11):
        new_tokens +=['colspan="'+str(i)+'"']
        new_tokens +=['rowspan="'+str(i)+'"']
    
    processor.tokenizer.add_tokens(new_tokens, special_tokens = False)
    
    #Configura o modelo
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<table_extraction>"])[0]
    model.generation_config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<table_extraction>"])[0]
    
    return model, processor

#Função de avaliação do modelo
def eval_model(model, processor, dataloader):
    out_dics = {}
    sum_score = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    for i, batch in enumerate(tqdm(dataloader)):
        
        pixel_values = batch["pixel_values"].to(device)
        filenames = batch["file_name"]
        gts = batch['gt']
        
        # autoregressively generate sequence
        outputs = model.generate(
            pixel_values,
            max_length= 1500,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams= 1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            )
        
        
        for sequence, filename in zip(outputs.sequences, filenames):
            table_html = "<html><body><table>" + processor.decode(sequence[2:-1]) + "</table></body></html>"
            print(table_html)
            out_dics[filename] = table_html
    return out_dics



#Carrega o conjunto de dados
import json
with open('../../aux/data/anns/val/val_dic.json') as fp:
    annotations = json.load(fp)

test_set = DonutTableDataset(annotations, 2048)

test_dataloader = DataLoader(test_set, batch_size=4, num_workers=4, shuffle=False)


models_dir = "../../aux/models/by_step/Pos_Enc/"

model_paths = [models_dir+model_path for model_path in os.listdir(models_dir)]

model_proc_pairs = [
                    ("../../aux/models/by_step/Pos_Enc/model_Pos-STEP_30000", "../../aux/processors/Donut_PubTables_HTML_Processor8k"),
]

for model_path in model_paths:
    model_proc_pairs.append(
                    (model_path, "../../aux/processors/Donut_PubTables_HTML_Processor8k")
    )



for model_path, proc_path in model_proc_pairs:
    model, processor = load_model_n_processor(model_path, proc_path)
    
    evals = eval_model(model, processor, test_dataloader)

    with open('../../aux/outputs/Pos_Enc/'+model_path.split('/')[-1]+'-output.json','w') as out:
        json.dump(evals, out)

