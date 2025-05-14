# 3D Topological Positioning in Generative Table Recognition

![GitHub](https://img.shields.io/badge/license-MIT-blue) ![GitHub top language](https://img.shields.io/github/languages/top/joaopaulo7/3D-Positioning-in-Generative-Table-Recognition)

This repository contains the implementation and evaluation code for our paper **"Evaluating Three-Dimensional Topological Positioning in Generative Table Recognition"**. It enables researchers to replicate our experiments and results.

![Screenshot from 2025-05-14 10-16-33](https://github.com/user-attachments/assets/91511f71-004e-4e5b-97c6-4855e7e1d618)


## ⚙️ Requirements
- Linux environment
- 4+ GPUs with 32GB+ VRAM (for training)
- 1 GPU with 32GB+ VRAM (for evaluation)
- Python 3.11+

## 🛠️ Setup Instructions

### 1. Download and Preprocess Data
Run the following scripts to download and preprocess the data:

```bash
./get_web_data.sh
./preprocess_data.sh
```
The preprocessing script will automatically train tokenizers and set up the required directory structure.

## 🚀 Training Models

Navigate to the scripts directory and run the appropriate training script:

```bash
cd proj/scripts
# Choose one or more of the following based on your model of interest
python3 Train_Model-3D_TML.py      # 3D Emb. TML model
python3 Train_Model-Pos_HTML.py    # Pos. Enc. HTML model
python3 Train_Model-3D_HTML.py     # 3D Emb. HTML model
python3 Train_Model-Pos_TML.py     # Pos. Enc. TML model
```

**Note:** All training scripts require at least 4 GPUs with 32GB+ VRAM.

## 📊 Evaluation

### 1. Generate Model Outputs
Run the corresponding evaluation script:

```bash
cd proj/scripts
# Choose the appropriate evaluation scripts
python3 Evaluate_Model-3D_TML.py      # 3D Emb. TML model
python3 Evaluate_Model-Pos_HTML.py    # Pos. Enc. HTML model
python3 Evaluate_Model-3D_HTML.py     # 3D Emb. HTML model
python3 Evaluate_Model-Pos_TML.py     # Pos. Enc. TML model
```

**Note:** Evaluation requires at least 1 GPU with 32GB+ VRAM.

Evaluation outputs are saved in JSON format in the `aux/outputs/` directory, organized by model and number of training steps:
```
aux/outputs/
├── [model_name]/
│   ├── [model]-[step]-output.json 
...
```

### 2. TEDS Evaluation
To compute Tree Edit Distance Scores, clone PubTabNet's, for the TEDS script, repository and run the evaluation:

```bash
cd aux
git clone https://github.com/ibm-aur-nlp/PubTabNet
python3 Eval_Outputs.py
```
**Note:** The Eval_Outputs.py script will automatically produce the scores for all files the generated in the previous step.

## 📂 Results

Evaluation scores are saved in JSON format under:
```
aux/outputs/
├── [model_name]/
│   └── evals/
│       ├── [model]-[step]-output-all.json       # Complete evaluation
│       └── [model]-[step]-output-struct.json    # Structure-only evaluation
│       ...
...
```

Example structure:
```
aux/outputs/3D_TML/evals/
├── model_3D_TML-STEP_12000-output-all.json
├── model_3D_TML-STEP_12000-output-struct.json
├── ...
└── model_3D_TML-3_EPOCHS-output-struct.json
```
