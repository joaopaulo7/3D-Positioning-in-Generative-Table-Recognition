import os
import sys
import json

sys.path.insert(0, 'PubTabNet/src')
from metric import TEDS


n_jobs = 8
teds_all = TEDS(n_jobs=n_jobs, ignore_nodes = "b", structure_only = False)
teds_struct = TEDS(n_jobs=n_jobs, ignore_nodes = "b", structure_only = True)

with open("data/anns/test/final_eval.json") as in_file:
    gt = json.load(in_file)


outputs_dirs = ["outputs/3D_Emb", "outputs/Pos_Enc"]
outputs = []

for output_dir in outputs_dirs:
    json_list = os.listdir(output_dir)
    for json_file in json_list:
        with open(output_dir+"/"+json_file) as in_file:
            outputs.append((json.load(in_file), output_dir, json_file[:-5]))


evaluations = {}

for output, output_dir, filename in outputs:
    scores_all = teds_all.batch_evaluate(output, gt)
    scores_struct = teds_struct.batch_evaluate(output, gt)
    
    evaluations[output_dir+"-evals/"+filename+"-all.json"] = scores_all
    evaluations[output_dir+"-evals/"+filename+"-struct.json"] = scores_struct

for outputs_dir in outputs_dirs:
    os.system("mkdir "+outputs_dir+"-evals")

for path, scores in evaluations.items():
    with open(path, "w") as out_file:
        json.dump(scores, out_file, indent=4)
