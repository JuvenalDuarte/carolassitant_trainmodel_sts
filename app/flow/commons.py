import luigi

luigi.interface.InterfaceLogging.setup(luigi.interface.core())

import os
import logging
import traceback

logger = logging.getLogger(__name__)

from pycarol.pipeline import Task
from luigi import Parameter
from datetime import datetime
from pycarol import Carol
from pycarol.apps import Apps

PROJECT_PATH = os.getcwd()
TARGET_PATH = os.path.join(PROJECT_PATH, 'luigi_targets')
Task.TARGET_DIR = TARGET_PATH
#Change here to save targets locally.
Task.is_cloud_target = True
Task.version = Parameter()
Task.resources = {'cpu': 1}

now = datetime.now()
now_str = now.isoformat()
_settings = Apps(Carol()).get_settings()

# The STS model to perform the fine tune. Currently are supported only the 
# models available on Sentence Transformers library, available on the link below:
# https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
baseline = _settings.get('baseline_model')

# The records in this table should be in the form:
#     {sentence 1} , {sentence 2} , {1/0}
# Where:
#     - The two sentences that should either be considered similar or not similar
#     - The third column indicates the sentences should be similar (1) or not similar (0)
training_table = _settings.get('training_pairs_table')

# Dictionary like mapping to define which column holds which information.
# Example: {"sentence1":"article_title", "sentence2":"ticket_title", "similarity":"cos_sim"}
# Notice: validation database, if provided, is suposed to follow the same mapping as the 
# training table (columns should be the same).
training_mapping = _settings.get('training_mapping')

# Similar to "training_pairs_table", but these records will be used to evaluate if
# the fine tunnning resulted in accuracy improvements compared to the baseline model 
# (without fine tuning).
validation_table = _settings.get('validation_pairs_table')

# TODO.
knowledgebase_file = _settings.get('knowledgebase_file')


# TODO.
unsupervised_finetune = _settings.get('unsupervised_finetune')

# This factor determines how agressive should the fine tunning/ domain adaptation be.
# If the factor is close to 1 then there's a high chance the pretrained model will fully 
# overwriten. When close to 0 it indicates the original model will preserved, but only
# small adjustments will be made by the training data.
finetune_factor = _settings.get('finetune_factor')

# For how many epochs will the fine tune be executed
# The longer the epochs, the more pre-trained knowledge will be overwritten
epochs = _settings.get('finetune_epochs')

# How many records will be feedforward and backpropagated per cycle.
# For memory saving use small batches, larger batches tend to lead to better results
# but may lead to memory overflow.
batchsize = _settings.get('finetune_batchsize')

# After fine tuning the model will be deployed to the app defined below, if the
# criteria is accepted. If list, model will be published to each of the apps
app_to_publish = _settings.get('app_to_publish')

# Allows user to specify condutions when the model is published after training.
# The options are given below:
#   - "test": app will not be published, can be used to evaluate training.
#   - "anyway": CAUTION! When this value is set the model will be published despites worse 
#               performance.
#   - "1" to "100": Defines an acceptable percent of improvement compared to the baseline 
#                to deploy the model.
#   - "0.0" to "1.0": Minimum acceptable accuracy on validation
publication_criteria = _settings.get('publication_criteria')

# TODO DO
reuse_ranking = _settings.get('reuse_ranking')

@Task.event_handler(luigi.Event.FAILURE)
def mourn_failure(task, exception):
    """Will be called directly after a failed execution
       of `run` on any JobTask subclass
    """
    logger.error(f'Error msg: {exception} ---- Error: Task {task},')
    traceback_str = ''.join(traceback.format_tb(exception.__traceback__))
    logger.error(traceback_str)


@Task.event_handler(luigi.Event.PROCESSING_TIME)
def print_execution_time(self, processing_time):
    logger.debug(f'### PROCESSING TIME {processing_time}s. Output saved at {self.output().path}')


#######################################################################################################

params = dict(
    version=os.environ.get('CAROLAPPVERSION', 'dev'),
    datetime = now_str,

    baseline = baseline,
    training_table = training_table,
    training_mapping = training_mapping,
    validation_table = validation_table,
    knowledgebase_file = knowledgebase_file,
    finetune_factor = finetune_factor,
    unsupervised_finetune = unsupervised_finetune,
    app_to_publish = app_to_publish,
    publication_criteria = publication_criteria,
    epochs = epochs,
    reuse_ranking = reuse_ranking,
    batchsize= batchsize
)
