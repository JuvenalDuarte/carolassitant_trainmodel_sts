from . import prepare_data
from . import load_model
from .commons import Task

from ..functions.run_baseline import run_baseline

import luigi
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

@inherit_list(
    load_model.LoadModel,
    prepare_data.PrepareData
)
class RunBaseline(Task):
    datetime = luigi.Parameter()
    baseline = luigi.Parameter()
    reuse_ranking = luigi.Parameter()
    ranking_train_strategy = luigi.Parameter()
    knowledgebase_file = luigi.Parameter()
    
    #target_type = PickleTarget
    #target_type = PytorchTarget

    def easy_run(self, inputs):
        model = inputs[0]
        train, validation = inputs[1]

        base_df = run_baseline(model=model,
                               model_name=self.baseline,
                               df_train=train, 
                               df_kb=self.knowledgebase_file,
                               reuse_ranking=self.reuse_ranking,
                               train_strat=self.ranking_train_strategy)

        return base_df
