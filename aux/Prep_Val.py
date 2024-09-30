import sys
import json
import os

ANN_PATH = 'data/anns/val/'

json_list = os.listdir(ANN_PATH)

html_list = []

for f in json_list:
    if(f[-6:-5] == "L"):
        html_list.append(f)

val_dic = {}

for file_name in html_list:
    with open(ANN_PATH + file_name, encoding="utf-8") as f:
        
        image_file = file_name.split('-')[0]+'.png'
        annotation = '<html><body><table>' + json.load(f) + '</table></body></html>'
        if 'colspan' in annotation or 'rowspan' in annotation:
            t_type = 'complex'
        else:
            t_type = 'simple'
        
        val_dic[image_file] = {'html': annotation, 'type': t_type}

with open(ANN_PATH+"val_dic.json", 'w') as out:
    json.dump(val_dic, out, ensure_ascii=False)