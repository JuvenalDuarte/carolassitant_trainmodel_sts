import logging
from sentence_transformers import util
from ..functions.run_baseline import getEmbeddingsCache
from torch.nn import MSELoss

logger = logging.getLogger(__name__)

def evaluate(baseline_model, tuned_model, df_val):

    logger.info(f'4. Evaluating performance.')

    uniq_sentences = list(df_val["sentence1"].unique())
    uniq_sentences = uniq_sentences + list(df_val["sentence2"].unique())
    uniq_sentences = list(set(uniq_sentences))
    total = len(uniq_sentences)
    logger.info(f'Processing {total} unique sentences on validation set.')


    logger.info(f'Calculating embeddings for baseline.')
    sentence2embedding = getEmbeddingsCache(uniq_sentences, baseline_model, cache=False)
    sentence1_embd_base = [sentence2embedding[s] for s in df_val["sentence1"].values]
    sentence2_embd_base = [sentence2embedding[s] for s in df_val["sentence2"].values]

    logger.info(f'Calculating embeddings for tuned.')
    sentence2embedding = getEmbeddingsCache(uniq_sentences, tuned_model, cache=False)
    sentence1_embd_tuned = [sentence2embedding[s] for s in df_val["sentence1"].values]
    sentence2_embd_tuned = [sentence2embedding[s] for s in df_val["sentence2"].values]


    logger.info(f'Calculating baseline similarities.')
    similarities = util.pytorch_cos_sim(sentence1_embd_base, sentence2_embd_base)
    df_val["baseline_similarity"] = similarities

    logger.info(f'Calculating tuned similarities.')
    similarities = util.pytorch_cos_sim(sentence1_embd_tuned, sentence2_embd_tuned)
    df_val["tuned_similarity"] = similarities


    loss = MSELoss()
    mse_baseline = loss(df_val["similarity"].values, df_val["baseline_similarity"].values)
    logger.info(f'Mean Squared Error (MSE) for baseline model: {mse_baseline}.')

    loss = MSELoss()
    mse_tuned = loss(df_val["similarity"].values, df_val["tuned_similarity"].values)
    logger.info(f'Mean Squared Error (MSE) for tuned model: {mse_tuned}.')
    
    return df_val