import logging
from sentence_transformers import util
from pycarol import Carol, Storage
import torch
import numpy as np

logger = logging.getLogger(__name__)

def getEmbeddingsCache(uniq_sentences, model, model_name, cache=True):

    # Loads the dictionary containing all sentences whose embeddings are already calculated.
    if cache:
        login = Carol()
        storage = Storage(login)
        filename = model_name + "_cache.pickle"

        logger.info('Loading cache from storage.')
        sent2embd = storage.load(filename, format='pickle', cache=False)

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
    
    else:
        logger.info('All sentences already present on cache, no calculation needed.')

    if cache:
        storage.save(filename, sent2embd, format='pickle')
    
    return sent2embd


def calculateSimilarities(embd_vec_1, embd_vec_2):
    #similarities = util.pytorch_cos_sim(embd_vec_1, embd_vec_2)

    # Calculating similarities individually to avoid memory overflow on 
    # matrix multiplication
    similarities = []
    for a, b in zip(embd_vec_1, embd_vec_2):
        similarities.append(float(util.pytorch_cos_sim(a, b)))

    return similarities

def run_baseline(model, model_name, df_train, df_kb):

    logger.info(f'2. Running baseline evaluation.')

    uniq_sentences = list(df_train["sentence1"].unique())
    uniq_sentences = uniq_sentences + list(df_train["sentence2"].unique())
    uniq_sentences = list(set(uniq_sentences))
    total = len(uniq_sentences)
    logger.info(f'Calculating embeddings for {total} unique sentences on training set.')
    sentence2embedding = getEmbeddingsCache(uniq_sentences, model, model_name, cache=True)

    logger.info(f'Translating sentences to embeddings.')
    sentence1_embd = [sentence2embedding[s] for s in df_train["sentence1"].values]
    sentence2_embd = [sentence2embedding[s] for s in df_train["sentence2"].values]
    df_train["search_embd"] = sentence2_embd

    logger.info(f'Calculating baseline similarities.')
    similarities = calculateSimilarities(sentence1_embd, sentence2_embd)
    df_train["baseline_similarity"] = similarities

    if df_kb:
        logger.info(f'Parsing \"knowledgebase_file\" setting.')

        login = Carol()
        kb_list = df_kb.split("/")
        if len(kb_list) == 4:
            kb_org, kb_env, kb_app, kb_file = kb_list
            login.switch_environment(org_name=kb_org, env_name=kb_env, app_name=kb_app)
        if len(kb_list) == 3:
            kb_env, kb_app, kb_file = kb_list
            login.switch_environment(org_name=login.organization, env_name=kb_env, app_name=kb_app)
        elif len(kb_list) == 2:
            kb_app, kb_file = kb_list
            login.switch_environment(org_name=login.organization, env_name=login.environment, app_name=kb_app)
        else:
            raise "Unable to parse \"knowledgebase_file\" setting. Valid options are: 1. org/env/app/file; 2. env/app/file; 3. app/file."

        storage = Storage(login)
        logger.info('Loading knowledge base from \"{df_kb}\".')
        df_kb = storage.load(kb_file, format='pickle', cache=False)

        logger.info('Calculating rankings. Articles on knowledge base: \"{df_kb.shape[0]}\".')
        filter_column = "module"
        for m in df_train[filter_column].unique():
            logger.info(f'Evaluating searchs on \"{m}\".')
            tmp1 = df_train[df_train[filter_column] == m].copy()
            tmp2 = df_kb[df_kb[filter_column] == m]

            targets = list(tmp1["article_id"].values)

            msg_embd_tensor = tmp1["search_embd"].values
            doc_embd_tensor = tmp2["sentence_embedding"].values

            score = util.pytorch_cos_sim(msg_embd_tensor, doc_embd_tensor)
            values, idx = torch.topk(score, k=1, dim=1, sorted=True)

    return df_train
