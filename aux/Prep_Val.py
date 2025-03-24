import sys
import json
import os
from multiprocessing import Pool


ANN_PATH = 'data/anns/val/'

json_list = os.listdir(ANN_PATH)

html_list = []

for f in json_list:
    if(f[-6:-5] == "L"):
        html_list.append(ANN_PATH+f)


    
def get_ann_dic(filename):
    with open(filename, encoding="utf-8") as f:
        
        image_file = filename.split('/val/')[1].split('-')[0]+'.png'
        annotation = '<html><body><table>' + json.load(f) + '</table></body></html>'
        if 'colspan' in annotation or 'rowspan' in annotation:
            t_type = 'complex'
        else:
            t_type = 'simple'
        
        return (image_file, {'html': annotation, 'type': t_type})


with Pool(processes = 12) as pool:
    res_dic_list = pool.map(get_ann_dic, html_list)


val_dic = {image_file: annotation for image_file, annotation in res_dic_list}


with open(ANN_PATH+"val_dic.json", 'w') as out:
    json.dump(val_dic, out, ensure_ascii=False)
