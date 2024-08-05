"""
Script for retrieving unlabeled instances that are close (or distant) to a given class of data. The instances are sorted
according to the distance. The script turns the texts into Sentence-BERT embeddings and then calculates the distances.

Demonstration of usage:
/notebooks/textgen_demo.ipynb

Experimental code:
/notebooks/semi_supervised_sbert.ipynb

"""

from sentence_transformers import SentenceTransformer, models
import numpy as np


def __get_centroid(embeddings):
    """
    function for getting the centroid of embeddings
    :param embeddings: embeddings of the data
    :return: centroid of the given embeddings
    """
    embedding_sum = np.sum(embeddings, axis=0)
    return np.divide(embedding_sum, len(embeddings[0]))


def __get_distances(embeddings, centroid, cosine=True):
    """
    pytorch based distance function that calculates the cosine or euclidean distance of all embeddings to the centroid
    :param embeddings: embeddings
    :param centroid: centroid of the embeddings from which the distance is calculated
    :param cosine: stating whether to use cosine or euclidean distance
    :return: distances of the embeddings to the centroid
    """
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    if cosine:
        return [cosine_similarity(embedding, centroid) for embedding in embeddings]
    else:
        return - [np.linalg.norm(centroid - embedding, 2) for embedding in embeddings]


def __get_k_nearest_indices(distances, k, inverse=False):
    """
    function for retrieving the k nearest indices of the distances
    :param distances: distances that were calculated before
    :param k: k
    :param inverse: boolean indicating if the sorted indices should be reversed
    :return: returns the k nearest indices of distances
    """
    distances = np.array(distances)
    if not inverse:
        return np.argsort(distances)[::-1][:k]
    else:
        return np.argsort(distances)[:k]


def __get_nearest_indices_threshold(distances, threshold, inverse=False):
    """
    function for retrieving the nearest indices of the distances from a given threshold
    :param distances: distances that were calculated before
    :param threshold: distance threshold from which the data should be included
    :param inverse: boolean indicating if all the distances from this threshold shoud be removed and if the sorted
        indices should be reversed
    :return: returns the nearest indices of the distances from a given threshold
    """
    if not inverse:
        distances = np.array(distances)
        threshold_count = len(list(filter(lambda x: x > threshold, distances)))
        return np.argsort(distances)[::-1][:threshold_count]
    else:
        distances = np.array(distances)
        threshold_count = len(list(filter(lambda x: x < threshold, distances)))
        return np.argsort(distances)[:threshold_count]


def apply_filtering(reference_data, unlabeled_data, close_instances=False,
                    sentence_transformer="roberta-large-nli-stsb-mean-tokens", quantile_threshold=0.15, verbose=False):
    """
    function that applies the filtering, which returns the unlabeled instances (from unlabeled_data) that are close
    (or distant @see :param distant_instances) to the centroid of the reference_data. The instances are sorted
    according to the distance
    :param reference_data: the data set from which the unlabeled data should be compared
    :param unlabeled_data: the data that should be labeled
    :param close_instances: stating weather distant or close instances should be filtered
    :param sentence_transformer: transformer model @see https://github.com/UKPLab/sentence-transformers for more models
    :param quantile_threshold: defining the quantile of the initial reference data that should be removed if
        the reference_data would be used in comparison with itself. The interpretation behind this is that some examples
        of the reference data are too close to the decision border, which should not define the distance threshold
    :param verbose: having several print statements
    :return: returns the unlabeled instances that are close (or distant) to the reference_data (sorted by their
        distance)
    """
    verbose_print = print if verbose else lambda *a, **k: None

    verbose_print("# Loading the sentence transformer #")
    model = SentenceTransformer(sentence_transformer)

    verbose_print("# Encoding reference data #")
    reference_data_embeddings = model.encode(np.array(reference_data))
    verbose_print("# Encoding unlabeled data #")
    unlabeled_data_embeddings = model.encode(np.array(unlabeled_data))

    verbose_print("# Calculating the exclusion boundary #")
    reference_data_centroid = __get_centroid(reference_data_embeddings)
    unlabeled_data_distances = __get_distances(unlabeled_data_embeddings, reference_data_centroid)
    reference_data_distances = __get_distances(reference_data_embeddings, reference_data_centroid)

    distance_threshold = np.quantile(reference_data_distances, quantile_threshold)
    index_of_remaining_dataset = __get_nearest_indices_threshold(unlabeled_data_distances, distance_threshold,
                                                                 close_instances)

    remaining_data = [unlabeled_data[index] for index in index_of_remaining_dataset]

    return remaining_data
