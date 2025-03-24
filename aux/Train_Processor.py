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


#configure TLM processor
TML_processor = TabeleiroProcessor.from_pretrained("processors/donut-base")

TML_processor.cell_types = ["<cell>", "<col_header>", "<row_header>", "<row_and_col_header>"]
TML_processor.content_types = ["<content_row_and_col_header>", "<content_row_header>", "<content_col_header>", "<content>"]
for i in range(2):
    for j in range(2):
        for k in range(2):
            TML_processor.cell_types.append("<span_type=0" + str(i) + str(j) + str(k) + ">")
            TML_processor.cell_types.append("<span_type=1" + str(i) + str(j) + str(k) + ">")

TML_processor_contents = ["<table_extraction>", "<table>", "<row>"] + TML_processor.content_types + TML_processor.cell_types


#configure HTML processor
HTML_processor = DonutProcessor.from_pretrained("processors/donut-base")

HTML_processor_contents  = ["<table_extraction>"]
HTML_processor_contents += ["<thead>", "</thead>", "<tbody>", "</tbody>"]
HTML_processor_contents += ["<tr>", "</tr>", "<td>", "</td>"]

HTML_processor_contents += ["<td ", ">"]
for i in range(1, 11):
    HTML_processor_contents += ['colspan="'+str(i)+'"']
    HTML_processor_contents += ['rowspan="'+str(i)+'"']



#load data
from tqdm.auto import tqdm
from multiprocessing import Pool, Value
import time

def load_func(file_name):
    content = []
    with open("data/anns/train/"+file_name, encoding="utf-8") as f:
        annotation = json.load(f)
    for row in annotation['tables'][0]:
        for cell in row:
            content.append(cell['content'])
    return content

with Pool(processes = 8) as pool:
    start_time = time.time()
    res_content_list = pool.map(load_func, json_list)

#flatten list
content_list = [content for process_content in res_content_list for content in process_content]


#train TML processor
TML_processor.tokenizer = TML_processor.tokenizer.train_new_from_iterator(
    content_list,
    new_special_tokens = TML_processor_contents,
    length = len(content_list),
    vocab_size = 8192,
    show_progress = True)

TML_processor.save_pretrained("processors/Donut_PubTables_TML_Processor8k")


#train HTML processor
HTML_processor.tokenizer = HTML_processor.tokenizer.train_new_from_iterator(
    content_list,
    new_special_tokens = HTML_processor_contents,
    length = len(content_list),
    vocab_size = 8192,
    show_progress = True)


HTML_processor.save_pretrained("processors/Donut_PubTables_HTML_Processor8k")
