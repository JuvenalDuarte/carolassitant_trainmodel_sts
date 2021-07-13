from pycarol import Carol, DataModel, Staging, Apps, PwdAuth, Storage
import os
import logging
from datetime import datetime, timedelta
import gc
import pandas as pd
import pickle
from sentence_transformers import LoggingHandler, util

logger = logging.getLogger(__name__)

def save_object_to_storage(storage, obj, filename):
    logger.info(f'Saving {filename} to the storage.')

    with open(filename, "bw") as f:
        pickle.dump(obj, f)

    storage.save(filename, obj, format='pickle')

def get_file_from_storage(storage, filename):
    return storage.load(filename, format='pickle', cache=False)


def getEmbeddingsCache(uniq_sentences, model, cache=True):
    login = Carol()
    storage = Storage(login)
    
    # Loads the dictionary containing all sentences whose embeddings are already calculated.
    filename = "embeddings_cache.pickle"
    if cache:
        logger.info('Loading cache from storage.')
        sent2embd = get_file_from_storage(storage, filename)

        if sent2embd is None:
            logger.warn('Unable to load file from storage. Reseting cache.')
            sent2embd = {}            

    else:
        # WARN: Full embeddings calculation demands high resources consumption.
        # Make sure the VM instance defined on manifest.json is big enough
        # before running on reset mode.
        logger.warn('Reseting cache.')
        sent2embd = {}
    
    # Gets the list of sentences for which embeddings are not yet calculated
    sentences_processed = list(sent2embd.keys())
    sentences_pending = list(np.setdiff1d(uniq_sentences,sentences_processed))

    if len(sentences_pending) > 0:

        logger.info(f'Calculating embeddings for {len(sentences_pending)} unprocessed sentences.')
        embeddings_pending = model.encode(sentences_pending, convert_to_tensor=False)
        dict_pending = dict(zip(sentences_pending, embeddings_pending))
        
        logger.info('Updating cache on storage.')
        sent2embd.update(dict_pending)
        save_object_to_storage(storage, sent2embd, filename)
    
    else:
        logger.info('All sentences already present on cache, no calculation needed.')
    
    return sent2embd

def run_baseline(model, df_train):

    logger.info(f'2. Running baseline evaluation.')

    uniq_sentences = list(df_train["sentence1"].unique())
    uniq_sentences = uniq_sentences + list(df_train["sentence2"].unique())
    uniq_sentences = list(set(uniq_sentences))
    total = len(uniq_sentences)
    logger.info(f'Calculating embeddings for {total} unique sentences on training set.')
    sentence2embedding = getEmbeddingsCache(uniq_sentences, model, cache=False)
    sentence1_embd = [sentence2embedding[s] for s in df_train["sentence1"].values]
    sentence2_embd = [sentence2embedding[s] for s in df_train["sentence2"].values]

    logger.info(f'Calculating baseline similarities.')
    similarities = util.pytorch_cos_sim(sentence1_embd, sentence2_embd)
    df_train["baseline_similarity"] = similarities

    return df_train
