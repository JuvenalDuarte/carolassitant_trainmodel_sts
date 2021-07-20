import logging
from sentence_transformers import util
import torch

logger = logging.getLogger(__name__)

def getEmbeddingsCache(uniq_sentences, model, cache=True):
    logger.info(f'Calculating embeddings for {len(uniq_sentences)} unprocessed sentences.')
    embeddings_pending = model.encode(uniq_sentences, convert_to_tensor=False)
    sent2embd = dict(zip(uniq_sentences, embeddings_pending))
    return sent2embd

def calculateSimilarities(embd_vec_1, embd_vec_2):
    #similarities = util.pytorch_cos_sim(embd_vec_1, embd_vec_2)

    # Calculating similarities individually to avoid memory overflow on 
    # matrix multiplication
    similarities = []
    for a, b in zip(embd_vec_1, embd_vec_2):
        similarities.append(float(util.pytorch_cos_sim(a, b)))

    return similarities

def run_baseline(model, df_train):

    logger.info(f'2. Running baseline evaluation.')

    uniq_sentences = list(df_train["sentence1"].unique())
    uniq_sentences = uniq_sentences + list(df_train["sentence2"].unique())
    uniq_sentences = list(set(uniq_sentences))
    total = len(uniq_sentences)
    logger.info(f'Calculating embeddings for {total} unique sentences on training set.')
    sentence2embedding = getEmbeddingsCache(uniq_sentences, model, cache=False)

    logger.info(f'Translating sentences to embeddings.')
    sentence1_embd = [sentence2embedding[s] for s in df_train["sentence1"].values]
    sentence2_embd = [sentence2embedding[s] for s in df_train["sentence2"].values]

    logger.info(f'Calculating baseline similarities.')
    similarities = calculateSimilarities(sentence1_embd, sentence2_embd)
    df_train["baseline_similarity"] = similarities

    return df_train
