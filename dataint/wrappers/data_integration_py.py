# data integration python client

# external
import datetime
import logging
import os
import pandas as pd

# project
from dataint.wrappers import schema_matcher_wrapper as SMW
from dataint.wrappers.schema_matcher_wrapper import Status


class SchemaMatcher(object):
    """
        Attributes:
            datasets : list of the instances of MatcherDataset which correspond to available datasets on the server;
            dataset_summary : pandas dataframework summarizing information about available datasets;
            models : list of instances of MatcherModel which correspond to available models on the server;
            model_summary : pandas dataframework summarizing information about available models;

            _session : class instance of SchemaMatcherSession;
            _column_map : a dictionary with keys as column ids and values as their names;
            _ds_keys : list of keys of all available datasets in the repository on the server;
            _model_keys : list of keys of all available models in the repository on the server;
    """

    def __init__(self):
        """
        Initialize class instance of SchemaMatcher.
        """
        logging.info('Initialising schema matcher class object.')
        self._session = SMW.SchemaMatcherSession()

        # we make all keys internal and try to hide the fact of their existence from the user
        self._ds_keys = self._get_dataset_keys()   # list of keys of all available datasets
        self._model_keys = self._get_model_keys()  # list of keys of all available models

        self.datasets, self.dataset_summary, self._column_map = self._populate_datasets()
        self.models, self.model_summary = self._get_model_summary()

    def _refresh(self): # TODO: improve
        """
        Refresh class attributes especially after new models or datasets have been added or deleted.

        Returns: Nothing.

        """
        # first we update datasets since column_map is updated here
        self._refresh_datasets()
        # column_map is used to update models
        self._refresh_models()

    def _refresh_datasets(self): # TODO: improve
        """
        Refresh class attributes related to datasets.

        Returns: Nothing.

        """
        self._ds_keys = self._get_dataset_keys()  # list of keys of all available datasets
        self.datasets, self.dataset_summary, self._column_map = self._populate_datasets()

    def _refresh_models(self): # TODO: optimize
        """
        Refresh class attributes related to models.

        Returns: Nothing.

        """
        self._model_keys = self._get_model_keys()  # list of keys of all available models
        # column_map is used to update models
        self.models, self.model_summary = self._get_model_summary()

    def _refresh_dataset(self, matcher_ds):
        """
        Refresh class attributes for datasets.
        The changes affect only those datasets which have matcher_ds.id.

        Args:
            matcher_ds: MatcherDataset

        Returns: Nothing

        """
        # update list of datasets
        for idx, ds in enumerate(self.datasets):
            if ds.id == matcher_ds.id:
                if ds == matcher_ds: # no changes are needed!
                    break
                self.datasets[idx] = matcher_ds
                # column_map should not change!
                break
        # in dataset_summary only description can change
        rows = self.dataset_summary[self.dataset_summary.dataset_id == matcher_ds.id].index.tolist()
        self.dataset_summary.set_value(rows, 'description', matcher_ds.description)

    def _remove_dataset(self, matcher_ds):
        """
        Remove dataset from schema matcher attributes.
        The changes affect only those datasets which have matcher_ds.id.

        Args:
            matcher_ds: MatcherDataset

        Returns: Nothing

        """
        self._ds_keys.remove(matcher_ds.id)  # remove the key
        self.datasets = [ds for ds in self.datasets if ds.id != matcher_ds.id]
        # remove this dataset from dataset_summary
        self.dataset_summary = self.dataset_summary[self.dataset_summary.dataset_id != matcher_ds.id]
        # update column_map
        if self._column_map:
            self._column_map = {k: v for k, v in self._column_map.items()
                                         if k not in matcher_ds._column_map}

    def _remove_model(self, matcher_model):
        """
        Remove model from schema matcher attributes.
        The changes affect only those models which have matcher_model.id.

        Args:
            matcher_model: MatcherModel

        Returns: Nothing

        """
        self._model_keys.remove(matcher_model.id)  # remove the key
        self.models = [ds for ds in self.models if ds.id != matcher_model.id]
        # remove this dataset from dataset_summary
        self.model_summary = self.model_summary[self.model_summary.model_id != matcher_model.id]


    def _refresh_model(self, matcher_model):
        """
        Refresh class attributes related to models.
        The changes affect only those models which have matcher_model.id.

        Args:
            matcher_model: MatcherModel

        Returns: Nothing.

        """
        # update list of models
        for idx, mod in enumerate(self.models):
            if mod.id == matcher_model.id:
                if mod == matcher_model: # no changes are needed!!!
                    break
                self.models[idx] = matcher_model
                # column_map should not change!
                break
        # model_summary can change drastically
        rows = self.model_summary[self.model_summary.model_id == matcher_model.id].index.tolist()
        self.model_summary.set_value(rows, 'description', matcher_model.description)
        self.model_summary.set_value(rows, 'created', matcher_model.date_created)
        self.model_summary.set_value(rows, 'modified', matcher_model.date_modified)
        self.model_summary.set_value(rows, 'status', matcher_model.model_state.status)
        self.model_summary.set_value(rows, 'state_created', matcher_model.model_state.date_created)
        self.model_summary.set_value(rows, 'state_modified', matcher_model.model_state.date_modified)

    def __str__(self):
        return "<SchemaMatcherAt(" + str(self._session.uri) + ")>"

    def __repr__(self):
        return self.__str__()

    def _get_dataset_keys(self):
        """Obtain the list of keys of all available datasets."""
        return self._session.list_alldatasets()

    def _get_model_keys(self):
        """Obtain the list of keys of all available models."""
        return self._session.list_allmodels()

    def _populate_datasets(self):
        """
        This method creates a dictionary (dataset key, instance of MatcherDataset).
        It also collects info on all datasets and puts this information
        into a nice Pandas data framework with the following columns:
        ds_key, file name, description, date created, date modified, num rows, num cols.
        Finally, this methods creates a dictionary to map column ids to column names.

        Returns: dictionary of dataset ids, Pandas dataframe, dictionary of all column ids.

        """
        datasets = []
        dataset_summary = []
        headers = ['dataset_id', 'name', 'description', 'created', 'modified', "num_rows", "num_cols"]
        column_map = dict()

        for ds_key in self._ds_keys:
            dataset = SMW.MatcherDataset(self._session.list_dataset(ds_key), self)
            dataset_summary.append([ds_key, dataset.filename, dataset.description,
                                    dataset.date_created, dataset.date_modified,
                                    dataset.sample.shape[0], dataset.sample.shape[1]])
            column_map.update(dataset._column_map)
            datasets.append(dataset)

        dataset_summary = pd.DataFrame(dataset_summary)
        if not dataset_summary.empty: # rename columns if data frame is not empty
            dataset_summary.columns = headers
        return datasets, dataset_summary, column_map

    def _get_model_summary(self):
        """
        This method creates a dictionary (model key, instance of MatcherModel).
        It also collects info on all models and puts this information
        into a nice Pandas data framework with the following columns:
        model_key, description, date created, date modified, model status, its creation date and modification date.

        Returns: dictionary of model ids, Pandas dataframe with summary of all models.

        """
        model_summary = []
        models = []
        for key in self._model_keys:
            mod = SMW.MatcherModel(self._session.list_model(key), self, column_map=self._column_map)
            model_summary.append([key, mod.description, mod.date_created, mod.date_modified,
                                  mod.model_state.status, mod.model_state.date_created, mod.model_state.date_modified])
            models.append(mod)

        model_summary = pd.DataFrame(model_summary)
        if not model_summary.empty:  # rename columns if data frame is not empty
            model_summary.columns = ['model_id','description','created','modified',
                                     'status','state_created','state_modified']
        return models, model_summary


    def _get_model(self, model_key):
        """
        Get the model corresponding to the key.

        Args:
            model_key : key of the model at the Schema Matcher server

        Returns: Instance of class MatcherModel corresponding to the model key

        """
        return SMW.MatcherModel(self._session.list_model(model_key), self._column_map)

    def _get_dataset(self, dataset_key):
        """
        Get the dataset corresponding to the key.

        Args:
            dataset_key : key of the dataset at the Schema Matcher server

        Returns: Instance of class MatcherDataset corresponding to the model key

        """
        return SMW.MatcherDataset(self._session.list_dataset(dataset_key))

    def create_model(self, feature_config,
                       description="",
                       classes=["unknown"],
                       model_type="randomForest",
                       labels=None, cost_matrix=None,
                       resampling_strategy="ResampleToMean"):

        """
        Post a new model to the schema matcher server.
        Refresh SchemaMatcher instance to include the new model.

        Args:
             feature_config : dictionary
             description : string which describes the model to be posted
             classes : list of class names
             model_type : string
             labels : dictionary
             cost_matrix :
             resampling_strategy : string

        Returns: MatcherModel

        """
        resp_dict = self._session.post_model(feature_config,
                                             description, classes, model_type,
                                             labels, cost_matrix, resampling_strategy) # send APi request
        new_model = SMW.MatcherModel(resp_dict, self, self._column_map)
        self._refresh_models() # we need to update class attributes to include new model
        # self._model_keys
        # self.models
        # self.model_summary
        return new_model

    def create_dataset(self, file_path, description, type_map):
        """
        Upload a new dataset to the schema matcher server.
        Refresh SchemaMatcher instance to include the new dataset.

        Args:
            description: string which describes the dataset to be posted
            file_path: string which indicates the location of the dataset to be posted
            type_map: dictionary with type map for the dataset

        Returns: newly uploaded MatcherDataset

        """
        new_dataset = SMW.MatcherDataset(self._session.post_dataset(description, file_path, type_map),
                                         self)
        self._refresh_datasets() # we need to update class attributes to include new datasets
        # self._ds_keys
        # self.datasets
        # self._column_map
        # self.dataset_summary
        return new_dataset

    def train_models(self, wait=True):
        """
        Launch training for all models in the repository.
        Please be aware that schema matcher at the moment can handle only one model training at a time.
        So, running this method with wait=False might not launch training for all models.

        Args:
            wait: boolean indicator whether to wait for the training to finish.

        Returns: True if training for all models succeeded.

        """
        return all([model.train(wait) for model in self.models])

    def get_predictions_models(self, wait=True):
        """
        Launch prediction for all models for all datasets in the repository.
        Please be aware that schema matcher at the moment can handle only one model prediction at a time.
        So, running this method with wait=False might not perform prediction for all models.

        Args:
            wait: boolean indicator whether to wait for the training to finish.

        Returns: Nothing, model attributes get updated.

        """
        [model.get_predictions(wait) for model in self.models]

    def error_models(self):
        """
        Go through the list of models and return those which have errors.

        Returns: list of models which have error state.

        """
        return [model for model in self.models
                if model.model_state.status == Status.ERROR]


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.curdir)))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    dm = SchemaMatcher()
    print(dm.models)

    error_mods = dm.error_models()

    print(dm.dataset_summary)
    # print(dm.column_map)

