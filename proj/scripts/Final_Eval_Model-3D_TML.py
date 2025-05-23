import os
import sys
from tqdm.auto import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


sys.path.insert(0, '../src')
from modeling_tabeleiro import TabeleiroModel
from processing_tabeleiro import TabeleiroProcessor


IMG_PATH = "../../aux/data/imgs/final_eval/"

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
        return len(self.annotations_files)
    
    
    def __getitem__(self, idx):
        
        file_name = self.annotations_files[idx]
        
        gt = self.annotations[file_name]['html']
        
        image = Image.open(IMG_PATH + file_name)
        
        
        # inputs
        pixel_values = processor.image_processor(image, random_padding=False, return_tensors="pt").pixel_values.squeeze()
        
        
        encoding = dict(file_name = file_name,
                        pixel_values=pixel_values,
                        gt = gt)
        
        return encoding

def load_model_n_processor(model_path, processor_path):
    
    #Carrega
    model = TabeleiroModel.from_pretrained(model_path) 

    #Carrega e configura o processador
    processor = TabeleiroProcessor.from_pretrained(processor_path)
    processor.image_processor.size = model.encoder.config.image_size[::-1] # should be (width, height)
    processor.image_processor.do_align_long_axis = False
    
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
            max_length= 2048,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams= 3,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            )

        for sequence, filename in zip(outputs.sequences, filenames):
            sequence = sequence.to('cpu')
            try:
                sequence = sequence[:sequence.tolist().index(processor.tokenizer.pad_token_id)]
            except ValueError:
                pass
            seq = torch.cat((sequence, torch.Tensor([2, 2]).int()), 0)
            table = processor.token2ann(seq, 2)
            table_html = "<html><body><table>" + processor.table2html(table['tables'][0]) + "</table></body></html>"
            out_dics[filename] = table_html
    return out_dics



#Carrega o conjunto de dados
import json
with open('../../aux/data/anns/test/final_eval.json') as fp:
    annotations = json.load(fp)

test_set = DonutTableDataset(annotations, 2048)

test_dataloader = DataLoader(test_set, batch_size=4, num_workers=4, shuffle=False)

    
    
model, processor = load_model_n_processor("../../aux/models/model-3D_TML-3_EPOCHS", "../../aux/processors/Donut_PubTables_TML_Processor8k")

evals = eval_model(model, processor, test_dataloader)

with open('../../aux/outputs/3D_TML/3D_TML-final_eval-output.json', 'w') as out:
    json.dump(evals, out, ensure_ascii=False)

