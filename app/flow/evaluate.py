from app.functions.run_finetuning import run_finetuning
from ..functions.evaluate import evaluate
from .commons import Task
import luigi
from . import prepare_data
from . import load_model
from . import run_finetuning
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

@inherit_list(
    load_model.LoadModel,
    run_finetuning.RunFineTuning,
    prepare_data.PrepareData
)
class Evaluate(Task):
    app_to_publish = luigi.Parameter()
    publication_criteria = luigi.Parameter()
    datetime = luigi.Parameter() 
    
    def easy_run(self, inputs):
        bmodel = inputs[0]
        tmodel = inputs[1]
        #train = inputs[2]
        validation = inputs[3]

        df_val = evaluate(baseline_model=bmodel, tuned_model=tmodel, df_val=validation)

        return df_val
