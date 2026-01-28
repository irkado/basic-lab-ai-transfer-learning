# Basic Lab AI Project: Transfer Learning for Image Classification

**Team Members**: Ekaterina Nedeva, Iryna Dolenko, Yaroslav Narozhnyi

The goal of this project is to classify plant leaf images into healthy or disease categories using a
CNN. We apply transfer learning by leveraging a pre-trained deep CNN and adapting it to the
PlantVillage dataset.

The project was developed using **Python 3.12**.

---

## Project Setup

1. Ensure that python version 3.12 is installed.
2. Clone the repository:
```bash
git clone https://github.com/irkado/basic-lab-ai-transfer-learning.git
cd basic-lab-ai-transfer-learning
```
3. To avoid conflicts with dependencies project uses venv virtual environment.<br>
**To create the environment:**
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
**To exit environment**
```bash
deactivate
```
---
add new library to .venv envi
-> ```pip install scikit-learn```<br>
update requirements.txt
-> ```pip freeze > requirements.txt```

## Dataset Setup
Generate an API token from the kaggle website after logging in.
Then install kaggle: 
```bash 
python -m pip install kaggle
```

After installing kaggle run these commands to download the PlantVillage dataset:
```bash
export KAGGLE_API_TOKEN="YOUR_TOKEN_HERE"
```

If you decide to change the name of the directory (data in this case) make sure to edit the path in preprocessing.py
```bash
mkdir -p data
kaggle datasets download -d mohitsingh1804/plantvillage -p data --unzip
```



## Workload Distribution
**Yarik** - Data Preparation: Load and preprocess our dataset, perform normalization, resizing,
and data augmentation, and split data into training, validation, and test sets

**Kate** - Model Design and Training: Select the pre-trained CNN architecture, implement and
configure the classifier head; also fine-tune last layers and tune learning rates

**Ira** - Evaluation: Evaluate model performance using accuracy and loss, analyze confusion
matrix and class-wise performance; also plot training/validation curves and summarize the results.