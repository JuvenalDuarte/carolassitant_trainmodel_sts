from ..functions.evaluate_results import evaluate_models
from .commons import Task
import luigi
from . import prepare_data
from . import load_model
from . import run_finetuning
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

#@inherit_list(
#    load_model.LoadModel,
#    run_finetuning.RunFineTuning,
#    prepare_data.PrepareData
#)
class EvaluateResults(Task):
    app_to_publish = luigi.Parameter()
    publication_criteria = luigi.Parameter()
    datetime = luigi.Parameter() 
    
    def easy_run(self, inputs):
        bmodel = inputs[0]
        tmodel = inputs[1]
        train, validation = inputs[2]

        if validation:
            df_val = evaluate_models(baseline_model=bmodel, tuned_model=tmodel, df_val=validation)
        else:
            df_val = None

        return df_val
