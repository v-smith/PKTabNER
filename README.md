
# PK Table Cell Classifier and NER

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/v-smith/PKTabCTC_NER/blob/master/LICENCE) ![version](https://img.shields.io/badge/version-0.1.0-blue) 

[**About the Project**](#about-the-project) | [**Dataset**](#dataset) | [**Getting Started**](#getting-started-) | [**Usage**](#usage) | [**Licence**](#lincence) | [**Citation**](#citation)

## About the Project

This repository contains the following:
1. custom pipes and models to classify table cells from scientific publications, depending on whether they contain a pharmacokinetic (PK) parameter reference. 
2. It further contains custom pipes and models to perform Named Entity Recognition to find Pharmacokinetic Parameter spans within table cell text. 
3. A heuristic pipeline to find mentions of Pharmacokinetic parameter context (e.g. number of subjects, units, measure types etc.)


#### Project Structure

- The main code is found in the root of the repository (see Usage below for more information).

```
├── annotation guidelines # used by annotators for annotating data in this project
├── configs # config files for training and inference arguments. 
├── data # 
├── pkcell # code for data preprocessing, post-processing, patterns, and model classes.
├── scripts # scripts for CTC and NER model training and inference.
├── .gitignore
├── LICENCE
├── README.md
├── requirements.txt
└── setup.py
```

#### Built With

[![Python v3.10](https://img.shields.io/badge/python-v3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)


## Dataset

The annotated PKTableClassification (PKTC) corpus can be downloaded from [zenodo](https://zenodo.org/records/13884895). The data is available under an MIT licence. The code assumes data is located in the `data` folder. 

## Getting Started 

#### Installation

To clone the repo:

`git clone https://github.com/v-smith/PKTableCTC_NER`
    
To create a suitable environment:
- ```conda create --name PKTableCTC_NER python==3.10```
- `conda activate PKTableCTC_NER`
- `conda install pytorch torchvision torchaudio cudatoolkit=12.6 -c pytorch`
- `pip install -e .`

#### GPU Support

Using a GPU is recommended for faster training. Single-GPU training has been tested with:

- `NVIDIA® GeForce RTX 2070`
- `CUDA 12.6`

## Usage

#### Train the CTC xgb pipeline:

````bash
python scripts/ctc/train_xgb_ctc.py
````

#### Evaluate the CTC xgb pipeline: 

````bash
python scripts/ctc/evaluate_xgb_ctc.py
````

#### Train the CTC BERT pipeline:

````bash
python scripts/ctc/train_adapted_bert_ctc.py
````

#### Evaluate the CTC BERT pipeline: 

```bash
python evaluate_adapted_bert_ctc.py
```

#### Train the NER BERT pipeline:

````bash
python scripts/pk_ner/finetune_bert.py
````

#### Evaluate the NER BERT pipeline: 

```bash
python scripts/ber/evaluate_bert_ner.py
```

## License
The codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

[mit]: LICENCE

## Citation
tbc
```bibtex
```



