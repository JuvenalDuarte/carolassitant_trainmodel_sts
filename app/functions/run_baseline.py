import logging
from sentence_transformers import util
from pycarol import Carol, Storage
import torch
import numpy as np
import pandas as pd
import gc
import math

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
        storage.save(filename, sent2embd, format='pickle', cache=False)
    
    return sent2embd


def calculateSimilarities(embd_vec_1, embd_vec_2):
    #similarities = util.pytorch_cos_sim(embd_vec_1, embd_vec_2)

    # Calculating similarities individually to avoid memory overflow on 
    # matrix multiplication
    similarities = []
    for a, b in zip(embd_vec_1, embd_vec_2):
        similarities.append(float(util.pytorch_cos_sim(a, b)))

    return similarities

def getRanking(test_set, knowledgebase, filter_column = "module", max_rank=100):
    out = []

    for m in test_set[filter_column].unique():
        logger.info(f'Evaluating searchs on \"{m}\".')
        tmp1 = test_set[test_set[filter_column] == m].copy()
        tmp2 = knowledgebase[knowledgebase[filter_column] == m].copy()

        narticles = tmp2["id"].nunique()
        articlesOnMod = list(tmp2["id"].unique())
        articlesOnMod = [str(a) for a in articlesOnMod]

        logger.info(f"Searching \"{len(tmp1)}\" messages within \"{narticles}\" articles for module {m}.")

        if len(tmp2) < 1:
            logger.warn(f"Module {m} not found on articles.")
            continue

        logger.info(f"Validating the expected articles are available on module {m}.")
        tmp1["available_on_kb"] = tmp1["article_id"].apply(lambda x: True if str(x) in articlesOnMod else False)
        not_avail = tmp1[~tmp1["available_on_kb"]]

        if (len(not_avail) > 0):
            na_art = list(not_avail["article_id"].unique())
            logger.warn(f"The following expected articles are not available on knowledge base under {m}: {na_art}.")

        tmp1 = tmp1[tmp1["available_on_kb"]]
        if len(tmp1) < 1 and (math.isnan(m) or m is None):
            logger.warn(f"Could not find any sample for \"{m}\". Discarding samples.")
            continue

        if len(tmp1) > 5000:
            tsearches = len(tmp1)
            logger.warn(f"Too many searches on the module {m}: \"{tsearches}\". Breaking down the process to avoid Out of Memory error.")

        f2 = next(c for c in tmp2.columns if "embedding" in c)
        torch_l2 = [torch.from_numpy(v) for v in tmp2[f2].values]
        doc_embd_tensor = torch.stack(torch_l2, dim=0)

        id_column = list(tmp2.columns).index("id")
        sentence_column = list(tmp2.columns).index("sentence")
            
        max_reg_per_round = 5000
        nrounds = max(1, round(len(tmp1)/max_reg_per_round, 0))
        for dft in np.array_split(tmp1, nrounds):

            targets = list(dft["article_id"].values)
            post = dft.copy()
            targetRanking = [9999] * len(post)

            all_scores_above = [np.nan] * len(post)
            all_sentences_above = [""] * len(post)
            all_articles_above = [""] * len(post)
            matching_sentence = [""] * len(post)
            matching_score = [np.nan] * len(post)

            f1 = "search_embd"
            torch_l1 = [torch.from_numpy(v) for v in dft[f1].values]
            msg_embd_tensor = torch.stack(torch_l1, dim=0)
            
            # High enough to find the target
            topranking = len(tmp2)
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
                        all_articles_above[i] = list(set(articles_above))
                        targetRanking[i] = len(all_articles_above[i]) + 1
                        matching_sentence[i] = sents[j]
                        matching_score[i] = scores[j]
                        all_sentences_above[i] = sentences_above
                        all_scores_above[i] = [round(float(s), 2) for s in scores_above]
                        break
                        
                    # Stop adding articles above to save memory
                    elif(len(sentences_above) < max_rank):
                        sentences_above.append(str(sents[j])) 
                        articles_above.append(str(preds[j]))
                        scores_above.append(scores[j])

            post["target_ranking"] = targetRanking
            post["all_sentences_above"] = all_sentences_above
            post["all_articles_above"] = all_articles_above
            post["all_scores_above"] = all_scores_above
            post["matching_sentence"] = matching_sentence
            post["matching_score"] = matching_score

            out.append(post)

    df4 = pd.concat(out)
        
    return df4

def run_baseline(model, model_name, df_train, df_kb, reuse_ranking, train_strat, ranking_threshold):
    logger.info(f'2. Running baseline evaluation.')

    if ranking_threshold in [np.nan, None, ""]:
        ranking_threshold = -1
    else:
        logger.info(f'Using {ranking_threshold} as reference threshold.')

    uniq_sentences = list(df_train["search"].unique())
    if not df_kb: uniq_sentences = uniq_sentences + list(df_train["target"].unique())
    uniq_sentences = list(set(uniq_sentences))
    total = len(uniq_sentences)
    logger.info(f'Calculating embeddings for {total} unique sentences on training set.')
    sentence2embedding = getEmbeddingsCache(uniq_sentences, model, model_name, cache=True)

    logger.info(f'Translating sentences to embeddings.')
    search_embd = [sentence2embedding[s] for s in df_train["search"].values]
    if not df_kb: target_embd = [sentence2embedding[s] for s in df_train["target"].values]
    df_train["search_embd"] = search_embd

    logger.info(f'Calculating baseline similarities.')
    if not df_kb: 
        similarities = calculateSimilarities(target_embd, search_embd)
        df_train["baseline_similarity"] = similarities
        del similarities
        del target_embd
    else:
        df_train["baseline_similarity"] = -1
        df_train["target"] = ""

    del sentence2embedding
    del search_embd
    gc.collect()

    baseline_top1_percent = None
    baseline_top3_percent = None
    reuse_ranking = False
    if df_kb:

        login = Carol()
        stg = Storage(login)
        if reuse_ranking:
            logger.info(f'Loading ranking from previous execution.')
            try:
                df_train = stg.load("baseline_ranking", format='pickle', cache=False)
            except:
                logger.info(f'Unble to load ranking from prevoius execution. Re-runing.')
                df_train = None

        if (not reuse_ranking) or (df_train is None):
            logger.info(f'Parsing \"knowledgebase_file\" setting.')
            
            login_kb = Carol()
            kb_list = df_kb.split("/")
            if len(kb_list) == 4:
                kb_org, kb_env, kb_app, kb_file = kb_list
                login_kb.switch_environment(org_name=kb_org, env_name=kb_env, app_name=kb_app)
            if len(kb_list) == 3:
                kb_env, kb_app, kb_file = kb_list
                login_kb.switch_environment(org_name=login_kb.organization, env_name=kb_env, app_name=kb_app)
            elif len(kb_list) == 2:
                kb_app, kb_file = kb_list
                login_kb.app_name = kb_app
            else:
                raise "Unable to parse \"knowledgebase_file\" setting. Valid options are: 1. org/env/app/file; 2. env/app/file; 3. app/file."

            logger.info(f'Loading knowledge base from \"{df_kb}\".')
            stg_kb = Storage(login_kb)
            df_kb = stg_kb.load(kb_file, format='pickle', cache=False)

            logger.info(f'Calculating rankings. Articles on knowledge base: \"{df_kb.shape[0]}\".')

            df_train = getRanking(test_set=df_train, 
                                  knowledgebase=df_kb, 
                                  filter_column="module",
                                  max_rank=5)

            df_train.drop(columns=["search_embd"], inplace=True)
            del df_kb
            gc.collect()

            #logger.info(f'Saving rankings for future executions.')
            #login = Carol()
            #stg = Storage(login)
            #stg.save("baseline_ranking", df_train, format='pickle', cache=False)

        total_tests = df_train.shape[0]
        baseline_top1 = sum((df_train["target_ranking"] == 1) & (df_train["matching_score"] > ranking_threshold))
        baseline_top1_percent = round((baseline_top1/total_tests) * 100, 2)
        logger.info(f'Baseline accuracy for Top 1: {baseline_top1} out of {total_tests} ({baseline_top1_percent}).')

        baseline_top3 = sum((df_train["target_ranking"] <= 3) & (df_train["matching_score"] > ranking_threshold))
        baseline_top3_percent = round((baseline_top3/total_tests) * 100, 2)
        logger.info(f'Baseline accuracy for Top 3: {baseline_top3} out of {total_tests} ({baseline_top3_percent}).')


        if train_strat.lower() == ">5":
            logger.info('Fine tuning model on records ranked greater than 5 or below threshold.')

            df_onlypos = df_train[(df_train["target_ranking"] <= 5) & (df_train["matching_score"] < ranking_threshold)]
            df_train = df_train[df_train["target_ranking"] > 5]

        elif train_strat.lower() == ">3":
            logger.info('Fine tuning model on records ranked greater than 3 or below threshold.')
            df_onlypos = df_train[(df_train["target_ranking"] <= 3) & (df_train["matching_score"] < ranking_threshold)]
            df_train = df_train[df_train["target_ranking"] > 3]

        elif train_strat.lower() == ">1":
            logger.info('Fine tuning model on records ranked greater than 1 or below threshold.')
            df_onlypos = df_train[(df_train["target_ranking"] == 1) & (df_train["matching_score"] < ranking_threshold)]
            df_train = df_train[df_train["target_ranking"] > 1]

        else:
            train_strat = "all"
            logger.info('Fine tuning model on all records.')

            # This case is essentially equals to ">1", but records where the model correctly
            # predicted the expected article will be used as positive examples to reinforce
            # and adjust attention heads.
            df_onlypos = df_train[df_train["target_ranking"] == 1]
            df_train = df_train[df_train["target_ranking"] > 1]

        logger.info(f'Total positive extracted from correctly predicted: {df_onlypos.shape[0]}.')
        df_onlypos = df_onlypos[["search", "baseline_similarity", "matching_sentence", "matching_score"]].copy()    

        df_onlypos["baseline_similarity"] = df_onlypos["matching_score"]
        df_onlypos["similarity"] = 1
        df_onlypos["target"] = df_onlypos["matching_sentence"]
        df_onlypos.dropna(subset=["search", "target", "baseline_similarity", "similarity"], inplace=True)
        df_onlypos.drop(columns=["matching_sentence","matching_score"], inplace=True)

        pos_samples = df_train[["search", "baseline_similarity", "matching_sentence", "all_scores_above"]].copy()
        neg_samples = df_train[["search", "all_sentences_above", "all_scores_above"]].copy()

        logger.info('Preparing positive samples.')
        # Forces the positive sample to be the highest score for the search.
        pos_samples["highest_returned_score"] = pos_samples["all_scores_above"].apply(lambda x: x[0] if type(x) is list else np.nan)
        pos_samples["baseline_similarity"] = pos_samples[["highest_returned_score", "baseline_similarity"]].max(axis=1)

        pos_samples["similarity"] = 1
        pos_samples["target"] = pos_samples["matching_sentence"]
        pos_samples.dropna(subset=["search", "target", "baseline_similarity", "similarity"], inplace=True)
        pos_samples.drop(columns=["matching_sentence","all_scores_above", "highest_returned_score"], inplace=True)

        pos_samples = pd.concat([pos_samples, df_onlypos], ignore_index=True)

        logger.info(f'Total positive samples: {pos_samples.shape[0]}.')

        logger.info('Preparing negative samples.')

        neg_samples_to_use = 1
        neg_samples["all_sentences_above"] = neg_samples["all_sentences_above"].apply(lambda x: x[:neg_samples_to_use] if type(x) is list else np.nan)
        neg_samples["all_scores_above"] = neg_samples["all_scores_above"].apply(lambda x: x[:neg_samples_to_use] if type(x) is list else np.nan)

        logger.info('Expanding wrong matches.')
        neg_samples_p1 = neg_samples.explode(column="all_sentences_above")
        neg_samples_p2 = neg_samples.explode(column="all_scores_above")

        neg_samples_p1["all_scores_above"] = neg_samples_p2["all_scores_above"]
        del neg_samples_p2
        gc.collect()

        neg_samples = neg_samples_p1.copy()
        del neg_samples_p1
        gc.collect()

        neg_samples["target"] = neg_samples["all_sentences_above"]
        neg_samples["baseline_similarity"] = neg_samples["all_scores_above"]
        neg_samples["similarity"] = 0
        neg_samples.drop(columns=["all_sentences_above","all_scores_above"], inplace=True)

        neg_samples.dropna(subset=["search", "target", "baseline_similarity", "similarity"], inplace=True)
        logger.info(f'Total negative samples: {neg_samples.shape[0]}.')

        logger.info('Concatenating positive and negative samples.')
        df_train = pd.concat([pos_samples, neg_samples], ignore_index=True)

    logger.info(f'Filtering inconsistent training samples.')

    # Filtering out same search-traget combinations with different expectations
    df_train["baseline_similarity"] = df_train["baseline_similarity"].astype(float)
    df_train = df_train.groupby(["search", "target"])[["similarity", "baseline_similarity"]].max()
    df_train = df_train.reset_index()

    # Filtering out when search is exactly the same as target
    df_train = df_train[~(df_train["search"] == df_train["target"])]

    logger.info(f'Total training samples after cleaning: {df_train.shape[0]}.')

    logger.info(f'Baseline evaluation finished.')
    baseline_acc = {"top1":baseline_top1_percent, 
                    "top3":baseline_top3_percent}
                    
    return df_train, baseline_acc
