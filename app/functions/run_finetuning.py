import logging
from sentence_transformers import InputExample, losses
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
    dft = df.astype({"sentence1": "str",
                    "sentence2": "str",
                    "target_similarity": "float"})
    
                                   
    dft.dropna(axis=0, how="any", inplace=True)

    pos_samples = []
    for (a, t, s) in dft[["sentence1", "sentence2", "target_similarity"]].values:
        pos_samples.append(InputExample(texts=[a, t], label=s))

    return pos_samples

def save_object_to_storage(obj, filename):
    login = Carol()
    storage = Storage(login)

    logger.info(f'Saving {filename} to the storage.')

    with open(filename, "bw") as f:
        pickle.dump(obj, f)

    storage.save(filename, obj, format='pickle')


def run_finetuning(baseline_model, baseline_name, baseline_df, bump, epchs = 10, bsize = 90):

    logger.info(f'3. Running fine tuning.')
    
    logger.info(f'Setting target as the baseline similarity bumped on the right direction.')
    baseline_df["target_similarity"] = baseline_df.apply(lambda x: applyBump(x, bump))

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