import sys
sys.path.insert(1, 'PubTabNet/src/')
sys.path.insert(1, '../proj/src/')

from transformers import DonutProcessor
from processing_tabeleiro import TabeleiroProcessor
import json
import os


SPLIT = "train/"

ANN_PATH = 'data/anns/'


json_list = os.listdir(ANN_PATH + SPLIT)

aux_list = []
html_list = []

for f in json_list:
    if(f[-6:-5] == "L"):
        html_list.append(f)
    else:
        aux_list.append(f)
json_list = aux_list

TML_processor = TabeleiroProcessor.from_pretrained("processors/donut-base")
HTML_processor = DonutProcessor.from_pretrained("processors/donut-base")


from tqdm.auto import tqdm

content_list = []

for i, file_name in enumerate(tqdm(json_list)):
    with open(ANN_PATH + SPLIT + file_name, encoding="utf-8") as f:
        annotation = json.load(f)
    for row in annotation['tables'][0]:
        for cell in row:
            content_list.append(cell['content'])
       

TML_processor.tokenizer = TML_processor.tokenizer.train_new_from_iterator(
    content_list,
    length = len(content_list),
    vocab_size = 8000,
    show_progress = True)
    

TML_processor.save_pretrained("processors/Donut_PubTables_TML_Processor8k")
    

HTML_processor.tokenizer = HTML_processor.tokenizer.train_new_from_iterator(
    content_list,
    length = len(content_list),
    vocab_size = 8000,
    show_progress = True)


HTML_processor.save_pretrained("processors/Donut_PubTables_HTML_Processor8k")
