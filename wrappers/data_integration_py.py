# data integration python client

# external
import datetime
import logging
import os

import pandas as pd
# project
from wrappers import schema_matcher_wrapper as SMW


class SchemaMatcher(object):
    """
        Attributes:

    """

    def __init__(self):
        """
        Initialize
        """
        logging.info('Initialising schema matcher class object.')
        self.session = SMW.SchemaMatcherSession()

        self.ds_keys = self.get_dataset_keys()   # list of keys of all available datasets
        self.model_keys = self.get_model_keys()  # list of keys of all available models
        self.dataset, self.dataset_summary, self.column_map = self.populate_datasets()
        self.model, self.model_summary = self.get_model_summary()

    def __str__(self):
        return "<SchemaMatcherAt(" + str(self.session.uri) + ")>"

    def __repr__(self):
        return self.__str__()

    def get_dataset_keys(self):
        """Obtain the list of keys of all available datasets."""
        return self.session.list_alldatasets()

    def get_model_keys(self):
        """Obtain the list of keys of all available models."""
        return self.session.list_allmodels()

    def populate_datasets(self):
        """
        Collect info on all datasets.

        :return: Pandas dataframe
        """
        dataset = dict()
        dataset_summary = []
        headers = ['dataset_id', 'name', 'description', 'created', 'modified', "num_rows", "num_cols"]
        column_map = dict()
        for ds_key in self.ds_keys:
            dataset[ds_key] = SMW.MatcherDataset(self.session.list_dataset(ds_key))
            dataset_summary.append([ds_key, dataset[ds_key].filename, dataset[ds_key].description,
                                    dataset[ds_key].date_created, dataset[ds_key].date_modified,
                                    dataset[ds_key].sample.shape[0], dataset[ds_key].sample.shape[1]])
            column_map.update(dataset[ds_key].column_map)
        dataset_summary = pd.DataFrame(dataset_summary)
        dataset_summary.columns = headers
        return dataset, dataset_summary, column_map

    def get_model_summary(self):
        """
        Collect info on all models.

        :return: Pandas dataframe with summary of all models
        """
        model_summary = []
        model = dict()
        for key in self.model_keys:
            mod = SMW.MatcherModel(self.session.list_model(key), self.column_map)
            model[key] = mod
            model_summary.append([key, mod.description, mod.date_created, mod.date_modified,
                                  mod.model_state.status, mod.model_state.date_created, mod.model_state.date_modified])

        model_summary = pd.DataFrame(model_summary)
        model_summary.columns = ['model_id','description','created','modified','status','state_created','state_modified']
        return model, model_summary


    def get_model(self, model_key):
        return SMW.MatcherModel(self.session.list_model(model_key), self.column_map)

    def create_model(self, feature_config,
                       description="",
                       classes=["unknown"],
                       model_type="randomForest",
                       labels=None, cost_matrix=None,
                       resampling_strategy="ResampleToMean"):

        """
        Post a new model to the schema matcher server.
        Args:
             feature_config: dictionary
             description: string which describes the model to be posted
             classes: list of class names
             model_type: string
             labels: dictionary
             cost_matrix:
             resampling_strategy: string

        :return: MatcherModel
        """
        resp_dict = self.session.post_model(feature_config,
                       description, classes,model_type,
                       labels, cost_matrix,resampling_strategy)
        return SMW.MatcherModel(resp_dict, self.column_map)



if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    dm = SchemaMatcher()
    dm.model

