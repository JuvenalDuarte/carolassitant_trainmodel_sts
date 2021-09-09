import logging
from sentence_transformers import util
from ..functions.run_baseline import getEmbeddingsCache, calculateSimilarities, getRanking
import torch
from datetime import datetime
from pycarol import Carol, Storage
import pickle

logger = logging.getLogger(__name__)

def save_model_to_onlineapp(obj, targetapp, filename):
    login = Carol()
    login.app_name = targetapp

    storage = Storage(login)

    logger.info(f'Saving {filename} to the app {targetapp}.')

    with open(filename, "bw") as f:
        pickle.dump(obj, f)

    storage.save(filename, obj, format='pickle')

def evaluate_models(baseline_name, target_app, baseline_model, tuned_model, df_val, df_kb):

    logger.info(f'4. Evaluating performance.')

    if not df_kb:
        uniq_sentences = list(df_val["target"].unique())
        uniq_sentences = uniq_sentences + list(df_val["search"].unique())
        uniq_sentences = list(set(uniq_sentences))
        total = len(uniq_sentences)
        logger.info(f'Processing {total} unique sentences on validation set.')


        logger.info(f'Calculating embeddings for baseline.')
        sentence2embedding = getEmbeddingsCache(uniq_sentences, baseline_model, baseline_name, cache=True)
        target_embd_base = [sentence2embedding[s] for s in df_val["target"].values]
        search_embd_base = [sentence2embedding[s] for s in df_val["search"].values]

        logger.info(f'Calculating embeddings for tuned.')
        sentence2embedding = getEmbeddingsCache(uniq_sentences, tuned_model, model_name="", cache=False)
        target_embd_tuned = [sentence2embedding[s] for s in df_val["target"].values]
        search_embd_tuned = [sentence2embedding[s] for s in df_val["search"].values]
        df_val["search_embd"]  = search_embd_tuned

        logger.info(f'Calculating baseline similarities.')
        similarities = calculateSimilarities(target_embd_base, search_embd_base)
        df_val["baseline_similarity"] = similarities

        logger.info(f'Calculating tuned similarities.')
        similarities = calculateSimilarities(target_embd_tuned, search_embd_tuned)
        df_val["tuned_similarity"] = similarities


        loss = torch.nn.MSELoss()
        target_tensor = torch.Tensor(df_val["similarity"].values)
        result_tensor = torch.Tensor(df_val["baseline_similarity"].values)
        mse_baseline = loss(target_tensor, result_tensor)
        logger.info(f'Mean Squared Error (MSE) for baseline model: {mse_baseline}.')

        loss = torch.nn.MSELoss()
        target_tensor = torch.Tensor(df_val["similarity"].values)
        result_tensor = torch.Tensor(df_val["tuned_similarity"].values)
        mse_tuned = loss(target_tensor, result_tensor)
        logger.info(f'Mean Squared Error (MSE) for tuned model: {mse_tuned}.')
    
    else:

        logger.info(f'Parsing \"knowledgebase_file\" setting.')

        uniq_sentences = list(df_val["search"].unique())
        uniq_sentences = list(set(uniq_sentences))
        total = len(uniq_sentences)
        logger.info(f'Processing {total} unique sentences on validation set.')

        logger.info(f'Calculating embeddings for tuned.')
        sentence2embedding = getEmbeddingsCache(uniq_sentences, tuned_model, model_name="", cache=False)
        search_embd_tuned = [sentence2embedding[s] for s in df_val["search"].values]
        df_val["search_embd"]  = search_embd_tuned
        
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

        logger.info(f'Updating embeddings on the knowledge base with the fine tuned model.')

        uniq_sentences = list(df_kb["sentence"].unique())
        sentence2embedding = getEmbeddingsCache(uniq_sentences, tuned_model, model_name="", cache=False)
        df_kb["sentence_embedding"]  = [sentence2embedding[s] for s in df_kb["sentence"].values]

        logger.info(f'Calculating rankings for the new model.')

        df_val = getRanking(test_set=df_val, 
                                knowledgebase=df_kb, 
                                filter_column="module",
                                max_rank=10)

        total_tests = df_val.shape[0]
        finetuned_top1 = sum(df_val["target_ranking"] == 1)
        finetuned_top1_percent = round((finetuned_top1/total_tests) * 100, 2)
        logger.info(f'Fine tuned accuracy for Top 1: {finetuned_top1} out of {total_tests} ({finetuned_top1_percent}).')

        finetuned_top3 = sum(df_val["target_ranking"] <= 3)
        finetuned_top3_percent = round((finetuned_top3/total_tests) * 100, 2)
        logger.info(f'Fine tuned accuracy for Top 3: {finetuned_top3} out of {total_tests} ({finetuned_top3_percent}).')


    if (target_app != ""):
        logger.info(f'Tunned model performed better than the baselina. Saving it to target app.')

        # Sends the model back to the CPU to asure compatibility with other apps
        tuned_model.to('cpu')

        saveto = baseline_name + "_FT" + str(datetime.now().strftime('%y%m%d%H%M%S'))
        for a in target_app.split(","):
            #Saves the new trained model to the target app.
            
            logger.info(f'Saving model to {saveto} on storage.')
            save_model_to_onlineapp(tuned_model, a, saveto)
    
    return df_val