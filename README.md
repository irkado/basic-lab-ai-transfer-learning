# Basic Lab AI Project: Transfer Learning for Image Classification

**Team Members**: Ekaterina Nedeva, Iryna Dolenko, Yaroslav Narozhnyi

The goal of this project is to classify plant leaf images into healthy or disease categories using a
CNN. We apply transfer learning by leveraging a pre-trained deep CNN and adapting it to the
PlantVillage dataset.

The project was developed using **Python 3.12**.

---

## Project Setup

Ensure that python version 3.12 is installed. 
Clone the repository:
```bash
git clone https://github.com/irkado/basic-lab-ai-transfer-learning.git
cd basic-lab-ai-transfer-learning
```
To avoid conflicts with dependencies, this project uses venv.<br>

**To create the environment, run:**
```bash
python3.12 -m venv .venv
```
**Activate the environment:**
```bash
source .venv/bin/activate
```
**Install dependencies (.venv must be activated):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**To exit the environment, run:**
```bash
deactivate
```
---
add a new library to .venv
-> ```pip install scikit-learn```<br>
update requirements.txt
-> ```pip freeze > requirements.txt```

---

## Dataset Setup
Go to kaggle.com and log in. Then go to Settings -> API Tokens -> Generate New Token. Open a terminal and run:
```bash
export KAGGLE_API_TOKEN=UR_TOKEN
```
This sets the token as an environment variable (Linux/MacOS).
To make it permanent just add it to ur shell config with:

```bash
echo 'export KAGGLE_API_TOKEN=UR_TOKEN' >> ~/.bashrc
#or ~/.zshrc or whatever shell
```
Reload and check if the token is active:
```bash
source ~/.bashrc
echo $KAGGLE_API_TOKEN
```

Then install kaggle: 
```bash 
python -m pip install kaggle
```

Download the PlantVillage dataset:
```bash
mkdir -p data
kaggle datasets download -d mohitsingh1804/plantvillage -p data --unzip
```

If you decide to change the name of the directory (`data` in this case) make sure to edit the path in preprocessing.py

---

## Training
To train the model, run:
```bash
python training.py --backbone densenet121
python training.py --backbone efficientnetb0
```
The trained model weights will be saved as ```{backbone}_best_head.pt``` and ```{backbone}_best_finetuned.pt```.

The results from evaluation (computed during training) will be saved as ```{backbone}_head_history.csv``` and ```{backbone}_finetuned_history.csv```.

---

## Evaluation and Inference
For the evaluation analysis, refer to ```evaluation.ipynb```. 

Inference is done at the end of the ```evaluation.ipynb``` notebook.

## Workload Distribution
**Yarik** - Data Preparation: Load and preprocess our dataset, perform normalization, resizing,
and data augmentation, and split data into training, validation, and test sets

**Kate** - Model Design and Training: Select the pre-trained CNN architecture, implement and
configure the classifier head; also fine-tune last layers and tune learning rates

**Ira** - Evaluation: Evaluate model performance using accuracy and loss, analyze confusion
matrix and class-wise performance; also plot training/validation curves and summarize the results.