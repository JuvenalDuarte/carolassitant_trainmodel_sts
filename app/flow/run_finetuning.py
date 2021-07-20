from . import run_baseline
from . import load_model
from .commons import Task
from ..functions.run_finetuning import run_finetuning
import luigi
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

@inherit_list(
    load_model.LoadModel,
    run_baseline.RunBaseline
)
class RunFineTuning(Task):
    baseline = luigi.Parameter() 
    finetune_factor = luigi.FloatParameter()
    epochs = luigi.IntParameter()
    batchsize = luigi.IntParameter()
    datetime = luigi.Parameter() 
    
    def easy_run(self, inputs):
        model = inputs[0]
        train_baseline = inputs[1]

        tuned_model = run_finetuning(baseline_model=model, baseline_name=self.baseline, baseline_df=train_baseline, bump=self.finetune_factor, epchs=self.epochs, bsize=self.batchsize)

        return tuned_model
