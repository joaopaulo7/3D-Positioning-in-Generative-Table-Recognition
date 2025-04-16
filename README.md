## This is the repository for the paper Evaluating Three-Dimensional Topological Positioning in Generative Table Recognition

---

This repository goal is to allow other researchers to replicate our results. The following steps show how one could do just that.

---

To train and evaluate models, you must first download web and preprocess data. Then, create axuliary directories and train new tokenizers.
In linux that can be done atomatically by calling two of the provided scripts:
```bash
./get_web_data.sh
./preprocess_data.sh
```

---

Next, to train models, run the training script, dependeing on which model you are currently working with. All of them require at least 4 GPUs with 32GB+ VRAM:
```bash
cd proj/sripts
# depending on which one you want to train
python3 Train_Model-3D_TML.py
# or
python3 Train_Model-Pos_HTML.py
# or
python3 Train_Model-3D_HTML.py
# or 
python3 Train_Model-Pos_TML.py
```

To generate validation set ouputs, run the evaluation script, dependeing on which model you are currently working with. All of them require at least one GPU with 32GB+ VRAM:
```bash
cd proj/sripts
# depending on which one you want to train
python3 Evaluate_Model-3D_TML.py
# or
python3 Evaluate_Model-Pos_HTML.py
# or
python3 Evaluate_Model-3D_HTML.py
# or 
python3 Evaluate_Model-Pos_TML.py
```

---

Finally, to evaluate outputs by Tree Edit Distance Score (TEDS):

You must first clone [PubTabNet's repository](https://github.com/ibm-aur-nlp/PubTabNet) to aux and then run evaluation script
```bash
cd aux
git clone https://github.com/ibm-aur-nlp/PubTabNet
python3 Eval_Outputs.py
```

---

By the end, the scores will be saved in json format in the aux/outputs/<model_name>/evals directories.

Scores are organized by training steps and type of evaluation(structure only or all):

```bash
aux/outputs/
├── 3D_HTML
├── 3D_TML
│   └── evals
│       ├── model_3D_TML-3_EPOCHS-output-all.json
│       ├── model_3D_TML-3_EPOCHS-output-struct.json
│       ├── model_3D_TML-STEP_12000-output-all.json
│       ├── model_3D_TML-STEP_12000-output-struct.json
│       ├── model_3D_TML-STEP_24000-output-all.json
│       ├── model_3D_TML-STEP_24000-output-struct.json
│       ├── model_3D_TML-STEP_36000-output-all.json
│       ├── model_3D_TML-STEP_36000-output-struct.json
│       ├── model_3D_TML-STEP_48000-output-all.json
│       ├── model_3D_TML-STEP_48000-output-struct.json
│       ├── model_3D_TML-STEP_60000-output-all.json
│       ├── model_3D_TML-STEP_60000-output-struct.json
│       ├── model_3D_TML-STEP_72000-output-all.json
│       └── model_3D_TML-STEP_72000-output-struct.json
│

```
