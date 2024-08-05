"""
A script for augmenting a data set with GPT-2. The whole augmentation process consists of several steps.
GPT-2 will be finetuned on the positive instances of the training data. All these instances will start with a special
start-of-data-token. The titles and this start-of-data-token will then be used as prefix for the generation of new
instances.
After this process it is advised to call bert_filtering.py to filter out any instances that do not represent the
original class well enough.

Demonstration of usage:
/notebooks/textgen_demo.ipynb

Experimental code:
/notebooks/textgeneration_gpt.ipynb
"""

import zipfile
import os
import re
import gpt_2_simple as gpt2
import tensorflow as tf


def get_zipped_dataset(path, set, exclude=".json"):
    """
    Function for retrieving the documents of a dataset stored in path dependent on set
    :param path: the path where the dataset is located
    :param set: stating which subfolder should be retrieved - meaning if test or training set should be retrieved
    :param exclude: a parameter for excluding files that contain the exclude-string
    :return:
    """

    def get_dataset_class(fn):
        """
        Nested function for getting the class of the file at hand
        :param fn: filename
        :return: 1 if document is positive and 0 if it is negative
        """
        split_name = fn.split(".")
        if len(split_name) > 1 and split_name[1].replace("U", "") == str(1):
            return 1
        else:
            return 0

    dataset_pos, dataset_neg, dataset_pos_titles, dataset_neg_titles = [], [], [], []

    with zipfile.ZipFile(path) as z:
        for filename in z.namelist():
            if set in filename and ".txt" in filename and exclude not in filename:
                with z.open(filename) as f:
                    full_text = f.read().decode('utf-8')
                    title = re.search('xxtitle (.*) xxbodytext', full_text).group(1)

                    if get_dataset_class(filename):
                        dataset_pos.append(full_text)
                        dataset_pos_titles.append(title)
                    else:
                        dataset_neg.append(full_text)
                        dataset_neg_titles.append(title)

    return dataset_pos, dataset_neg, dataset_pos_titles, dataset_neg_titles


def __prepare_finetuning_file(path, data):
    prefix = "<|startoftext|> "
    suffix = " <|endoftext|>"

    file_path = path + "finetuning.txt"
    file = open(file_path, "w")
    for text in data:
        file.write(prefix)
        file.write(text)
        file.write(suffix)
        file.write("\n\n")
    file.close()

    return file_path


def __finetune(path_of_dataset, positive_train_data, model_name="335M", steps=1000, overwrite=False, verbose=False):
    """
    function for finetuning the GPT-2 model with the positive training data
    :param path_of_dataset: path of the dataset
    :param positive_train_data: the positive instances of the training data set
    :param model_name: the GPT-2 model that should be used for generation.
        "124M": small model
        "355M": medium model
        "774M": large model (finetuning may fail - not tested)
        "1558M": extra large model (may not work at all)
    :param steps: steps indicate how long the GPT-2 model should be finetuned. It is advisable to run the model till
        the average loss is about 0.2. If the default 2000 steps are note sufficient you can run this function again
        to train it further (the saved model will be loaded if another call is performed, when :param overwrite is False
    :param overwrite: If True the saved model for this specific data set will be discarded. Otherwise the old model
        for this data set will be restored and the finetuning will be continued.
    :param verbose: having several print statements
    :return: the finetuned GPT-2 model and the tensorflow session
    """
    verbose_print = print if verbose else lambda *a, **k: None

    verbose_print("# Preparing the finetuning file #")
    file_path = __prepare_finetuning_file(path_of_dataset, positive_train_data)

    verbose_print("# Loading the model #")
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/*model_name*/

    verbose_print("# Starting the finetuning #")
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  file_path,
                  model_name=model_name,
                  run_name='run',
                  sample_every=1000,
                  steps=steps,
                  overwrite=overwrite)

    os.remove(file_path)

    return gpt2, sess


def __generate_text(gpt2, sess, model_name, positive_train_titles, path_of_dataset, temperature=0.7, nsamples=10,
                    title_token="xxtitle", body_token="xxbodytext", verbose=False):
    """
    function for generating text with the given GPT-2 model.
    :param gpt2: GPT-2 model
    :param sess: Tensorflow session
    :param model_name: name of the model
    :param positive_train_titles: titles of the positive training instances
    :param path_of_dataset: path of the data set
    :param temperature: temperature indicates how creative the generated text should be. The higher the more creative
        the texts will be. According to the authors a number between 0.7 and 1.0 should work  well
    :param nsamples: nsamples states how many instances per training instance should be generated
    :param title_token: token that proceeds the title
    :param body_token: token that proceeds the body
    :param verbose: having several print statements
    :return: the path of the generated instances
    """
    verbose_print = print if verbose else lambda *a, **k: None
    if not os.path.exists(path_of_dataset + "data/"):
        os.makedirs(path_of_dataset + "data/")
    if not os.path.exists(path_of_dataset + "data/generated_samples/"):
        os.makedirs(path_of_dataset + "data/generated_samples/")

    for index, title in enumerate(positive_train_titles):
        prefix_gen = "<|startoftext|> " + title_token + " " + title + " " + body_token
        gpt2.generate(sess,
                      model_name=model_name,
                      prefix=prefix_gen,
                      truncate="<|endoftext|>",
                      include_prefix=True,
                      destination_path=path_of_dataset + "data/generated_samples/generated_" + str(index) + ".txt",
                      temperature=temperature,
                      nsamples=nsamples)

        verbose_print(str(index + 1) + "/" + str(len(positive_train_titles)))

    return path_of_dataset + "generated_samples/"


def reading_generated_data(path_of_generated_data):
    """
    function for reading the generated data that is stored in path_of_generated_data
    :param path_of_generated_data: path where the generated data is stored
    :return: the generated data in form of a list (each text is one element)
    """
    index = 0
    new_data = []
    while os.path.isfile(path_of_generated_data + "/generated_" + str(index) + ".txt"):
        file_name = path_of_generated_data + "/generated_" + str(index) + ".txt"
        with open(file_name) as file:
            texts = file.read()
            temp_data = texts.split("\n====================\n<|startoftext|> ")
            temp_data[0] = temp_data[0].replace("<|startoftext|> ", "")
        new_data.extend(temp_data)
        index += 1

    return new_data


def apply_gpt(path_of_dataset, model_name="335M", steps=2000, overwrite=False, temperature=0.7, nsamples=10,
              title_token="xxtitle", body_token="xxbodytext", verbose=False):
    """
    function for generating data according to the GPT-2 data augmentation process (see description of the whole script).
    The positive class of the given data set will be augmented.

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
    """
    verbose_print = print if verbose else lambda *a, **k: None

    if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
        print("No GPU detected!")

    verbose_print("# Getting the data from the dataset #")
    positive_train_data, _, positive_train_titles, _ = get_zipped_dataset(
        path_of_dataset + "data.zip", "train", exclude=".json")

    gpt2, sess = __finetune(path_of_dataset, positive_train_data, model_name=model_name, steps=steps,
                            overwrite=overwrite, verbose=verbose)

    path_of_generated_data = __generate_text(gpt2, sess, model_name, positive_train_titles, path_of_dataset,
                                             temperature=temperature, nsamples=nsamples, title_token=title_token,
                                             body_token=body_token, verbose=verbose)

    generated_data = reading_generated_data(path_of_generated_data)

    return generated_data, positive_train_data, path_of_generated_data
