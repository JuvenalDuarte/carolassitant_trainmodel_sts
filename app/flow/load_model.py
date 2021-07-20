from .commons import Task
import luigi
from sentence_transformers import SentenceTransformer
from pycarol.pipeline.targets import PytorchTarget
import logging
#import torch

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

class LoadModel(Task):
    baseline = luigi.Parameter()
    target_type = PytorchTarget

    def easy_run(self, inputs):

#        try:
#            gpu = torch.cuda.is_available()
#            logger.info(f'GPU enabled? {gpu}.')
#            if gpu:
#                logger.info(f'GPU model: {torch.cuda.get_device_name(0)}.')
#        except Exception as e:
#            logger.error(f'Cannot verify if GPU is available: {e}.')

        logger.info(f'Loading baseline model: {self.baseline}.')
        model = SentenceTransformer(self.baseline)

        return model
