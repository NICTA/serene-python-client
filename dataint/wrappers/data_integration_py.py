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
            session : class instance of SchemaMatcherSession;
            ds_keys : list of keys of all available datasets in the repository on the server;
            model_keys : list of keys of all available models in the repository on the server;
            dataset : dictionary with keys as dataset keys and values as the corresponding instances of MatcherDataset;
            dataset_summary : pandas dataframework summarizing information about available datasets;
            column_map : a dictionary with keys as column ids and values as their names;
            model : dictionary with keys as model keys and values as the corresponding instances of MatcherModel;
            model_summary : pandas dataframework summarizing information about available models;
    """

    def __init__(self):
        """
        Initialize class instance of SchemaMatcher.
        """
        logging.info('Initialising schema matcher class object.')
        self.session = SMW.SchemaMatcherSession()

        self.ds_keys = self.get_dataset_keys()   # list of keys of all available datasets
        self.model_keys = self.get_model_keys()  # list of keys of all available models
        self.dataset, self.dataset_summary, self.column_map = self.populate_datasets()
        self.model, self.model_summary = self.get_model_summary()

    def refresh(self):
        """
        Refresh class attributes especially after new models or datasets have been added or deleted.
        Returns:
        """
        # first we update datasets since column_map is updated here
        self.refresh_datasets()
        # column_map is used to update models
        self.refresh_models()

    def refresh_datasets(self):
        """
        Refresh class attributes related to datasets.
        Returns:
        """
        self.ds_keys = self.get_dataset_keys()  # list of keys of all available datasets
        self.dataset, self.dataset_summary, self.column_map = self.populate_datasets()

    def refresh_models(self):
        """
        Refresh class attributes related to models.
        Returns:
        """
        self.model_keys = self.get_model_keys()  # list of keys of all available models
        # column_map is used to update models
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
        This method creates a dictionary (dataset key, instance of MatcherDataset).
        It also collects info on all datasets and puts this information
        into a nice Pandas data framework with the following columns:
        ds_key, file name, description, date created, date modified, num rows, num cols.
        Finally, this methods creates a dictionary to map column ids to column names.

        Returns: dictionary of dataset ids, Pandas dataframe, dictionary of all column ids.
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
        if not(dataset_summary.empty): # rename columns if data frame is not empty
            dataset_summary.columns = headers
        return dataset, dataset_summary, column_map

    def get_model_summary(self):
        """
        This method creates a dictionary (model key, instance of MatcherModel).
        It also collects info on all models and puts this information
        into a nice Pandas data framework with the following columns:
        model_key, description, date created, date modified, model status, its creation date and modification date.

        Returns: dictionary of model ids, Pandas dataframe with summary of all models.
        """
        model_summary = []
        model = dict()
        for key in self.model_keys:
            mod = SMW.MatcherModel(self.session.list_model(key), self.column_map)
            model[key] = mod
            model_summary.append([key, mod.description, mod.date_created, mod.date_modified,
                                  mod.model_state.status, mod.model_state.date_created, mod.model_state.date_modified])

        model_summary = pd.DataFrame(model_summary)
        if not (model_summary.empty):  # rename columns if data frame is not empty
            model_summary.columns = ['model_id','description','created','modified',
                                     'status','state_created','state_modified']
        return model, model_summary


    def get_model(self, model_key):
        """
        Get the model corresponding to the key.
        Args:
            model_key : key of the model at the Schema Matcher server

        Returns: Instance of class MatcherModel corresponding to the model key

        """
        return SMW.MatcherModel(self.session.list_model(model_key), self.column_map)

    def get_dataset(self, dataset_key):
        """
        Get the dataset corresponding to the key.
        Args:
            dataset_key : key of the dataset at the Schema Matcher server

        Returns: Instance of class MatcherDataset corresponding to the model key

        """
        return SMW.MatcherDataset(self.session.list_dataset(dataset_key))

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
        resp_dict = self.session.post_model(feature_config,
                       description, classes,model_type,
                       labels, cost_matrix,resampling_strategy) # send APi request
        new_model = SMW.MatcherModel(resp_dict, self.column_map)
        self.refresh_models() # we need to update class attributes to include new model
        return new_model

    def upload_dataset(self, file_path, description, type_map):
        """
        Upload a new dataset to the schema matcher server.
        Refresh SchemaMatcher instance to include the new dataset.
        Args:
            description: string which describes the dataset to be posted
            file_path: string which indicates the location of the dataset to be posted
            type_map: dictionary with type map for the dataset

        Returns: newly uploaded MatcherDataset

        """
        new_dataset = SMW.MatcherDataset(self.session.post_dataset(description, file_path, type_map))
        self.refresh_datasets() # we need to update class attributes to include new datasets
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
        return all([model.train(self.session, wait) for model in self.model.values()])

    def get_predictions_models(self, wait=True):
        """
        Launch prediction for all models for all datasets in the repository.
        Please be aware that schema matcher at the moment can handle only one model prediction at a time.
        So, running this method with wait=False might not perform prediction for all models.
        Args:
            wait: boolean indicator whether to wait for the training to finish.

        Returns: Nothing, model attributes get updated.
        """
        [model.get_predictions(self.session, wait) for model in self.model.values()]

    def error_models(self):
        """
        Go through the list of models and return those which have errors.
        Returns: list of models which have error state.
        """
        return dict([(key,model) for key,model in self.model.items()
                             if model.model_state.status == Status.ERROR])


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.curdir)))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    dm = SchemaMatcher()
    dm.model

    error_mods = dm.error_models()
    print(len(error_mods))
    print(dm.train_models())

    # print(dm.dataset_summary)
    # print(dm.column_map)

    model = list(dm.model.values())[0]
    # print(model.all_data)

    model.get_predictions(dm.session)
    snap = model.all_data[["column_id","actual_label","predicted_label"]].dropna()
    equalSnap = snap[snap["actual_label"] == snap["predicted_label"]]
    print(snap.shape)
    print(equalSnap.shape)
    print("Equal: " + str(snap.shape == equalSnap.shape))

