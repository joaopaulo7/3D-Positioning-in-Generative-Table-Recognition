import os
import sys
import json
from multiprocessing import Pool

sys.path.insert(0, 'PubTabNet/src')
from metric import TEDS


with open("data/anns/test/final_eval.json") as in_file:
    gt = json.load(in_file)


output_file = "outputs/3D_TML/3D_TML-final_eval-output.json"
eval_path   = "outputs/3D_TML/evals/3D_TML-final_eval"

with open(output_file) as in_file:
    output = json.load(in_file)
            

n_jobs = 48
teds_all = TEDS(n_jobs=n_jobs, ignore_nodes = "b", structure_only = False)
teds_struct = TEDS(n_jobs=n_jobs, ignore_nodes = "b", structure_only = True)

evaluations = {}

scores_all = teds_all.batch_evaluate(output, gt)
with open(eval_path+"-all.json", "w") as out_file:
    json.dump(scores_all, out_file, indent=4)


scores_struct = teds_struct.batch_evaluate(output, gt)
with open(eval_path+"-struct.json", "w") as out_file:
    json.dump(scores_all, out_file, indent=4)
