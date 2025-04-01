import os
import sys
import json
from multiprocessing import Pool

sys.path.insert(0, 'PubTabNet/src')
from metric import TEDS


with open("data/anns/val/val_dic.json") as in_file:
    gt = json.load(in_file)


outputs_dirs = ["outputs/3D_TML", "outputs/Pos_Enc"]
outputs = []

for outputs_dir in outputs_dirs:
    json_list = os.listdir(outputs_dir)
    for json_file in json_list:
        with open(outputs_dir+"/"+json_file) as in_file:
            outputs.append((json.load(in_file), outputs_dir, json_file[:-5]))
            


n_jobs = 48
teds_all = TEDS(n_jobs=n_jobs, structure_only = False)
teds_struct = TEDS(n_jobs=n_jobs, structure_only = True)

evaluations = {}

for output, output_dir, filename in outputs:
    for file in output:
        output[file] = output[file].replace('<td> ', '<td>')
    scores_all = teds_all.batch_evaluate(output, gt)
    scores_struct = teds_struct.batch_evaluate(output, gt)
    
    evaluations[output_dir+"/evals/"+filename+"-all.json"] = scores_all
    evaluations[output_dir+"/evals/"+filename+"-struct.json"] = scores_struct
    

for path, scores in evaluations.items():
    with open(path, "w") as out_file:
        json.dump(scores, out_file, indent=4)
