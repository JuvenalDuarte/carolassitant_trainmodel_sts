import logging
from sentence_transformers import InputExample, datasets, losses
from torch.utils.data import DataLoader
from pycarol import Carol, Storage
from datetime import datetime
import random
import math
import pickle

logger = logging.getLogger(__name__)

'''
Used to evaluate how many parameters are set to be trained.
'''
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

'''
Freezes all layers in the model except for the output dense layer.
'''
def freeze_layers(model):
    trainable_previous = count_parameters(model)
    current_layer = 0
    freeze_till = 0
    
    for child in model.children():
        if (current_layer <= freeze_till):
            print("INFO* Freezing layer %d." %(current_layer))
            
            for param in child.parameters():
                param.requires_grad = False
                
            current_layer += 1
        else:
            break
            
    trainable_after = count_parameters(model) 
    print("INFO* Trainable parameters reduced from %d to %d." %(trainable_previous, trainable_after))

def applyBump(row, bump):
    # when the sentences are labeled as similar, the similarity will be increased by bump %
    if row["similarity"] == 1:
        factor = 1 + bump

    # when the sentences are labeled as not similar, the similarity will be decreased by bump %
    else:
        factor = 1 - bump

    return row["baseline_similarity"] * factor

# Transforms records into sample layout required for STS training
def prepare_samples(df):
    dft = df.astype({"search": "str",
                    "target": "str",
                    "target_similarity": "float"})
    
                                   
    dft.dropna(subset=["search", "target", "target_similarity"], axis=0, how="any", inplace=True)

    pos_samples = []
    for (a, t, s) in dft[["search", "target", "target_similarity"]].values:
        pos_samples.append(InputExample(texts=[a, t], label=s))

    return pos_samples

def save_object_to_storage(obj, filename):
    login = Carol()
    storage = Storage(login)

    logger.info(f'Saving {filename} to the storage.')

    with open(filename, "bw") as f:
        pickle.dump(obj, f)

    storage.save(filename, obj, format='pickle')

def unsupervised_pretrain_TSDAE(baselinemodel, model_name, sentence_list, epochs=10):
    from nltk import download

    download('punkt')

    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(sentence_list)

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Use the denoising auto-encoder loss
    if model_name == "LaBSE":
        decoder_name = model_name
    else:
        decoder_name = 'bert-base-uncased'

    train_loss = losses.DenoisingAutoEncoderLoss(baselinemodel, decoder_name_or_path=decoder_name, tie_encoder_decoder=True)

    # Call the fit method
    baselinemodel.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=False
    )

    return baselinemodel

def run_finetuning(baseline_model, baseline_name, baseline_df, bump, unsup_pretrain, epchs = 10, bsize = 90):

    logger.info(f'3. Running fine tuning.')

    if unsup_pretrain == "TSDAE":
        logger.info(f'Running unsupervised pretraining through TSDAE. Using search and target fields as input text.')
        pretraintext = list(baseline_df["search"].values) + list(baseline_df["target"].values)
        baseline_model = unsupervised_pretrain_TSDAE(baseline_model, baseline_name, pretraintext, epochs=epchs)
    
    logger.info(f'Setting target as the baseline similarity bumped on the right direction.')
    baseline_df["target_similarity"] = baseline_df.apply(lambda x: applyBump(x, bump), axis=1)

    logger.info(f'Saving training data for reference.')
    login = Carol()
    stg = Storage(login)
    stg.save("training_samples", baseline_df, format='pickle')

    logger.info(f'Preparing the dataset.')
    samples_df = prepare_samples(baseline_df)

    #Data is originally ordered. To avoid similar tickets to be grouped on the same batches, 
    #we shuffle the data on this step.
    logger.info(f'Shuffling training data.')
    random.shuffle(samples_df)

    #logger.info(f'Freezing layers.')
    #freeze_layers(model)

    logger.info(f'Preparing batches. Batch size={bsize}.')
    train_dataloader = DataLoader(samples_df, shuffle=True, batch_size=bsize)

    logger.info(f'Fine tuning the model. Epochs={epchs}.')
    warmup_steps = math.ceil(len(train_dataloader) * epchs * 0.1)

    train_loss = losses.CosineSimilarityLoss(baseline_model)
    baseline_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epchs, warmup_steps=warmup_steps)

    logger.info(f'Fine tuning concluded.')

    #Saves the new trained model to the current dir.
    saveto=baseline_name + "_FT" + str(datetime.now().strftime('%y%m%d%H%M%S'))
    logger.info(f'Saving model to {saveto} on storage.')
    save_object_to_storage(baseline_model, saveto)
    
    return baseline_model