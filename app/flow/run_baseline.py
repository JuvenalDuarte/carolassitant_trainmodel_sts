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
    
    #target_type = PickleTarget
    #target_type = PytorchTarget

    def easy_run(self, inputs):
        model = inputs[0]
        train = inputs[1]
        #validation = inputs[2]

        base_df = run_baseline(model=model, df_train=train)

        return base_df
