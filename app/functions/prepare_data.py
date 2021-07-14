from pycarol import Carol, Staging
import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

def fetchFromCarol(conn, stag, carol=None, columns=None):
    if carol is None:
        carol = Carol()

    try:
        df = Staging(carol).fetch_parquet(staging_name=stag, connector_name=conn, backend='pandas', columns=columns, cds=True)

    except Exception as e:
        logger.error(f'Failed to fetch dada. {e}')
        df =  pd.DataFrame()

    return df

def prepare_data(training_table, validation_table, attr_map):

    logger.info(f'1. Preparing dataset for training.')

    logger.info(f'Parsing \"training_pairs_table\" setting.')

    # Parsing details about the training table connection parameters
    training_list = training_table.split("/")
    if len(training_list) == 3:
        train_env, train_conn, train_stag = training_list
    elif len(training_list) == 2:
        train_conn, train_stag = training_list
    else:
        raise "Unable to parse \"training_pairs_table\" setting. Valid options are: 1. env/connector/staging; 2. connector/staging."

    # Unable to handle the env option for now
    logger.info(f'Retrieving training data from {train_conn}/{train_stag}.')
    train_dataset = fetchFromCarol(conn=train_conn, stag=train_stag)


    # Parsing details about the validation table connection parameters
    # Note: validation table is optional
    if validation_table != "":
        logger.info(f'Parsing \"validation_pairs_table\" setting.')

        validation_list = validation_table.split("/")
        if len(validation_list) == 3:
            val_env, val_conn, val_stag = validation_list
        elif len(validation_list) == 2:
            val_conn, val_stag = validation_list
        else:
            raise "Unable to parse \"validation_pairs_table\" setting. Valid options are: 1. env/connector/staging; 2. connector/staging."

        logger.info(f'Retrieving validation data from {val_conn}/{val_stag}.')
        validation_dataset = fetchFromCarol(conn=train_conn, stag=train_stag)

    else:
        validation_dataset = pd.DataFrame()

    logger.info(f'Parsing \"training_mapping\" setting.')
    attr_map = json.loads(attr_map)

    similarity_col = attr_map["similarity"]
    total_training = len(train_dataset)
    total_validation = len(validation_dataset)
    similar_training = len(train_dataset[train_dataset[similarity_col] == 1])
    similar_ratio_training = 1.0 * similar_training/ total_training
    disimilar_ratio_training = 1.0 * (total_training - similar_training)/ total_training

    logger.info(f'Datasets summary:')
    logger.info(f' - Total training records: {total_training}.')
    logger.info(f' - Similar pairs (%) on training: {similar_ratio_training}.')
    logger.info(f' - Disimilar pairs (%) on training: {disimilar_ratio_training}.')
    logger.info(f' - Total validation records: {total_validation}.')

    logger.info(f'Setting standard column names')
    train_dataset.rename(columns=attr_map, inplace=True)
    validation_dataset.rename(columns=attr_map, inplace=True)

    logger.info(f'Filtering target columns only')
    train_dataset = train_dataset[attr_map.values()] 
    validation_dataset = validation_dataset[attr_map.values()]

    return train_dataset, validation_dataset
