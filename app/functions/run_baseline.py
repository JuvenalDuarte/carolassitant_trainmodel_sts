import logging
from sentence_transformers import util

logger = logging.getLogger(__name__)

def getEmbeddingsCache(uniq_sentences, model, cache=True):
    logger.info(f'Calculating embeddings for {len(uniq_sentences)} unprocessed sentences.')
    embeddings_pending = model.encode(uniq_sentences, convert_to_tensor=False)
    sent2embd = dict(zip(uniq_sentences, embeddings_pending))
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
