# Data Augmentation in Natural Language Processing

This repository contains the scripts and notebooks for the paper "Data augmentation in natural language processing: a novel text generation approach for long and short text classifiers" by Markus Bayer, Marc-André Kaufhold, Björn Buchhold, Marcel Keller, Jörg Dallmeyer, and Christian Reuter.

## Overview

The codebase provides tools for augmenting datasets in Natural Language Processing (NLP) tasks using the advanced model GPT-2. The augmentation process aims to improve the performance of text classifiers for both long and short texts by generating additional data and filtering it to ensure quality.

## Repository Structure

- `gpt_da.py`: Script for augmenting a dataset with GPT-2.
- `bert_filtering.py`: Script for filtering augmented data using Sentence-BERT.
- `notebooks/`: Directory containing demonstration and experimental notebooks.

## Scripts

The process consists of several steps:

1. Fine-tuning GPT-2 on the positive instances of the training data.
2. Using a special start-of-data token along with titles as prefixes for generating new instances.
3. (Recommended) Filtering the generated instances using `bert_filtering.py` to ensure they represent the original class well.

### gpt_da.py

This script performs data augmentation using GPT-2. 

#### Usage

```python
# Example usage in a Jupyter notebook
/notebooks/textgen_demo.ipynb

For augmenting the data run function apply_gpt():
    :param path_of_dataset: path of the dataset
    :param model_name: the GPT-2 model that should be used for generation.
        "124M": small model
        "355M": medium model
        "774M": large model (finetuning may fail - not tested)
        "1558M": extra large model (may not work at all)
    :param steps: steps indicate how long the GPT-2 model should be finetuned. It is advisable to run the model till
        the average loss is about 0.2. If the default 2000 steps are note sufficient you can run this function again
        to train it further (the saved model will be loaded if another call is performed, when :param overwrite is False
    :param overwrite: If True the saved model for this specific data set will be discarded. Otherwise the old model
        for this data set will be restored and the finetuning will be continued
    :param temperature: temperature indicates how creative the generated text should be. The higher the more creative
        the texts will be. According to the authors a number between 0.7 and 1.0 should work  well
    :param nsamples: nsamples states how many instances per training instance should be generated
    :param title_token: token that proceeds the title
    :param body_token: token that proceeds the body
    :param verbose: having several print statements
    :return: the generated data, the positive training data (important for the bert filtering that should be called
        afterwards), and the path of the generated data

```

#### Experimental Code

```python
# Experimental usage in a Jupyter notebook
/notebooks/textgeneration_gpt.ipynb
```

### bert_filtering.py

This script filters the augmented data by retrieving unlabeled instances that are either close to or distant from a given class. It uses Sentence-BERT embeddings to calculate the distances and sort the instances accordingly.

#### Usage

```python
# Example usage in a Jupyter notebook
/notebooks/textgen_demo.ipynb
```

#### Experimental Code

```python
# Experimental usage in a Jupyter notebook
/notebooks/semi_supervised_sbert.ipynb
```

## Notebooks

The `notebooks/` directory contains the following:

- `textgen_demo.ipynb`: Demonstrates the usage of `gpt_da.py` and `bert_filtering.py`.
- `textgeneration_gpt.ipynb`: Contains experimental code for text generation using GPT-2.
- `semi_supervised_sbert.ipynb`: Contains experimental code for semi-supervised learning using Sentence-BERT.

## Installation

To run the scripts and notebooks, you need to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions to this project. Please open an issue or submit a pull request with your changes.

## Contact

For any questions or inquiries, please contact the main author:

- Markus Bayer: [bayer@peasec.tu-darmstadt.de](mailto:bayer@peasec.tu-darmstadt.de)

---

Technical University of Darmstadt, Darmstadt, Germany  
CID GmbH, Freigericht, Germany  
Hanau, Germany
