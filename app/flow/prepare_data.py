from ..functions.prepare_data import prepare_data
from ..flow.commons import Task
import luigi
import pandas as pd
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

class PrepareData(Task):
    training_table = luigi.Parameter()
    validation_table = luigi.Parameter()
    training_mapping = luigi.Parameter()
    datetime = luigi.Parameter() 

    def easy_run(self, inputs):
        df_train, df_val = prepare_data(training_table=self.training_table, validation_table=self.validation_table, attr_map=self.training_mapping)
        return df_train, df_val