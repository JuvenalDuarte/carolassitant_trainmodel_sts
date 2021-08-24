import logging
from sentence_transformers import util
from pycarol import Carol, Storage
import torch
import numpy as np
import pandas as pd

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

def getRanking(test_set, knowledgebase, filter_column = "module"):
    out = []

    for m in test_set[filter_column].unique():
        logger.info(f'Evaluating searchs on \"{m}\".')
        tmp1 = test_set[test_set[filter_column] == m].copy()
        tmp2 = knowledgebase[knowledgebase[filter_column] == m].copy()

        narticles = tmp2["id"].nunique()

        logger.info(f"INFO* Searching \"{len(tmp1)}\" messages within \"{narticles}\" articles for module {m}.")

        if len(tmp1) < 1 and (math.isnan(m) or m is None):
            logger.warn(f"WARN* Module not define \"{m}\". Searching through all articles.")
            continue

        if len(tmp2) < 1:
            logger.warn(f"WARN* Module {m} not found on articles.")
            continue

        targets = list(tmp1["article_id"].values)

        post = tmp1.copy()
        targetRanking = [9999] * len(post)

        all_scores_above = [np.nan] * len(post)
        all_sentences_above = [""] * len(post)
        all_articles_above = [""] * len(post)
        matching_sentence = [""] * len(post)

        f1 = "search_embd"
        torch_l1 = [torch.from_numpy(v) for v in tmp1[f1].values]
        msg_embd_tensor = torch.stack(torch_l1, dim=0)
        
        f2 = "sentence_embedding"
        torch_l2 = [torch.from_numpy(v) for v in tmp2[f2].values]
        doc_embd_tensor = torch.stack(torch_l2, dim=0)
        
        id_column = list(tmp2.columns).index("id")
        sentence_column = list(tmp2.columns).index("sentence")
        topranking = min(len(tmp2), 1000)
        score = util.pytorch_cos_sim(msg_embd_tensor, doc_embd_tensor)
        values_rank, idx_rank = torch.topk(score, k=topranking, dim=1, sorted=True)

        for i in range(len(idx_rank)):
            preds = tmp2.iloc[idx_rank[i, :].tolist(), id_column]
            sents = tmp2.iloc[idx_rank[i, :].tolist(), sentence_column]
            scores = values_rank[i, :]

            if np.isscalar(preds):
                preds = [preds]
                sents = [sents]
            else:
                preds = list(preds.values)
                sents = list(sents.values)

            articles_above = []
            sentences_above = []
            scores_above = []
            for j in range(len(preds)):
                if str(preds[j]) == str(targets[i]):
                    # takes the highest ranking only, since an article is represented by multiple 
                    # sentences (title, tags, question)
                    targetRanking[i] = j + 1
                    matching_sentence[i] = sents[j]
                    all_articles_above[i] = ",".join(list(set(articles_above)))
                    all_sentences_above[i] = "|".join(sentences_above)
                    all_scores_above[i] = [round(float(s), 2) for s in scores_above]
                    break
                    
                else:
                   sentences_above.append(str(sents[j])) 
                   articles_above.append(str(preds[j]))
                   scores_above.append(scores[j])

        post["target_ranking"] = targetRanking
        post["all_sentences_above"] = all_sentences_above
        post["all_articles_above"] = all_articles_above
        post["all_scores_above"] = all_scores_above
        post["matching_sentence"] = matching_sentence

        out.append(post)

    df4 = pd.concat(out)
        
    return df4

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
            login.app_name = kb_app
        else:
            raise "Unable to parse \"knowledgebase_file\" setting. Valid options are: 1. org/env/app/file; 2. env/app/file; 3. app/file."

        storage = Storage(login)
        logger.info('Loading knowledge base from \"{df_kb}\".')
        df_kb = storage.load(kb_file, format='pickle', cache=False)

        logger.info('Calculating rankings. Articles on knowledge base: \"{df_kb.shape[0]}\".')

        rank_df = getRanking(test_set=df_train, knowledgebase=df_kb, filter_column="module")

        total_tests = df_train.shape[0]
        baseline_top1 = len(rank_df[rank_df["target_ranking"] == 1])
        baseline_top1_percent = round((baseline_top1/total_tests) * 100, 2)
        logger.info('Baseline accuracy for Top 1: {baseline_top1} out of {total_tests} ({baseline_top1_percent}).')

        baseline_top3 = len(rank_df[rank_df["target_ranking"] <= 3])
        baseline_top3_percent = round((baseline_top3/total_tests) * 100, 2)
        logger.info('Baseline accuracy for Top 3: {baseline_top3} out of {total_tests} ({baseline_top3_percent}).')

        no_training_needed = rank_df[rank_df["target_ranking"] <= 3]
        training_needed = rank_df[rank_df["target_ranking"] > 3]

        logger.info('Preparing positive samples.')
        pos_samples = training_needed.copy()
        pos_samples["baseline_similarity"] = pos_samples["all_scores_above"].apply(lambda x: x[0] if type(x) is list else np.nan)
        pos_samples["similarity"] = 1

        logger.info('Preparing negative samples.')
        neg_samples = training_needed.copy()
        neg_samples_to_use = 1
        neg_samples["all_sentences_above"] = pos_samples["all_sentences_above"].apply(lambda x: x[:neg_samples_to_use] if type(x) is list else np.nan)
        neg_samples["all_scores_above"] = pos_samples["all_scores_above"].apply(lambda x: x[:neg_samples_to_use] if type(x) is list else np.nan)

        neg_samples_p1 = neg_samples.explode(column="all_sentences_above")
        neg_samples_p2 = neg_samples.explode(column="all_scores_above")

        neg_samples_p1["all_scores_above"] = neg_samples_p2["all_scores_above"]

        neg_samples = neg_samples_p1.copy()
        neg_samples["sentence1"] = neg_samples["all_sentences_above"]
        neg_samples["baseline_similarity"] = neg_samples["all_scores_above"]
        neg_samples["similarity"] = 0

        df_train = pd.concat([pos_samples, neg_samples], ignore_index=True)

    return df_train
