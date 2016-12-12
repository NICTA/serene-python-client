"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Serene Python client: Data Integration Software
"""
import datetime
import logging
import os
import os.path
import pandas as pd
import collections
import pprint

from .utils import convert_datetime
from .wrappers import schema_matcher as api
#from .wrappers.schema_matcher import Status
from enum import Enum


class SchemaMatcher(object):
    """
        Attributes:
            datasets : list of the instances of MatcherDataset which correspond to available datasets on the server;
            dataset_summary : pandas dataframework summarizing information about available datasets;
            models : list of instances of MatcherModel which correspond to available models on the server;
            model_summary : pandas dataframework summarizing information about available models;

            _api : class instance of SchemaMatcherSession;
            _column_map : a dictionary with keys as column ids and values as their names;
            _ds_keys : list of keys of all available datasets in the repository on the server;
            _model_keys : list of keys of all available models in the repository on the server;
    """

    def __init__(self,
                 host="127.0.0.1",
                 port=8080,
                 auth=None,
                 cert=None,
                 trust_env=None):
        """
        Initialize class instance of SchemaMatcher.
        """
        logging.info('Initialising schema matcher class object.')
        self.api = api.Session(host, port, auth, cert, trust_env)

        # we make all keys internal and try to hide the fact of their existence from the user
        #self._ds_keys = self._get_dataset_keys()   # list of keys of all available datasets
        #self._model_keys = self._get_model_keys()  # list of keys of all available models

        #self.datasets, self.dataset_summary, self._column_map = self._populate_datasets()
        #self.models, self.model_summary = self._get_model_summary()

    @property
    def datasets(self):
        """Maintains a list of DataSet objects"""
        keys = self.api.dataset_keys()
        ds = DataSetList()
        for k in keys:
            ds.append(NewDataSet(self.api.dataset(k), self))
        return ds

    @property
    def models(self):
        """Maintains a list of Model objects"""
        keys = self.api.model_keys()
        ms = ModelList()
        for k in keys:
            ms.append(NewModel(self.api.model(k), self))
        return ms

    # def _refresh(self):
    #     """
    #     Refresh class attributes especially after new models or datasets have been added or deleted.
    #
    #     Returns: Nothing.
    #
    #     """
    #     # first we update datasets since column_map is updated here
    #     self._refresh_datasets()
    #     # column_map is used to update models
    #     self._refresh_models()
    #
    # def _refresh_datasets(self):
    #     """
    #     Refresh class attributes related to datasets.
    #
    #     Returns: Nothing.
    #
    #     """
    #     self._ds_keys = self._get_dataset_keys()  # list of keys of all available datasets
    #     self.datasets, self.dataset_summary, self._column_map = self._populate_datasets()
    #
    # def _refresh_models(self):
    #     """
    #     Refresh class attributes related to models.
    #
    #     Returns: Nothing.
    #
    #     """
    #     self._model_keys = self._get_model_keys()  # list of keys of all available models
    #     # column_map is used to update models
    #     self.models, self.model_summary = self._get_model_summary()
    #
    # def _refresh_dataset(self, matcher_ds):
    #     """
    #     Refresh class attributes for datasets.
    #     The changes affect only those datasets which have matcher_ds.id.
    #
    #     Args:
    #         matcher_ds: MatcherDataset
    #
    #     Returns: Nothing
    #
    #     """
    #     # update list of datasets
    #     for idx, ds in enumerate(self.datasets):
    #         if ds.id == matcher_ds.id:
    #             if ds == matcher_ds:  # no changes are needed!
    #                 break
    #             self.datasets[idx] = matcher_ds
    #             # column_map should not change!
    #             break
    #     # in dataset_summary only description can change
    #     rows = self.dataset_summary[self.dataset_summary.dataset_id == matcher_ds.id].index.tolist()
    #     self.dataset_summary.set_value(rows, 'description', matcher_ds.description)

    # def _remove_dataset(self, matcher_ds):
    #     """
    #     Remove dataset from schema matcher attributes.
    #     The changes affect only those datasets which have matcher_ds.id.
    #
    #     Args:
    #         matcher_ds: MatcherDataset
    #
    #     Returns: Nothing
    #
    #     """
    #     self._ds_keys.remove(matcher_ds.id)  # remove the key
    #     self.datasets = [ds for ds in self.datasets if ds.id != matcher_ds.id]
    #
    #     # remove this dataset from dataset_summary
    #     self.dataset_summary = self.dataset_summary[self.dataset_summary.dataset_id != matcher_ds.id]
    #
    #     # update column_map
    #     if self._column_map:
    #         self._column_map = {k: v for k, v in self._column_map.items()
    #                             if k not in matcher_ds._column_map}

    # def _remove_model(self, matcher_model):
    #     """
    #     Remove model from schema matcher attributes.
    #     The changes affect only those models which have matcher_model.id.
    #
    #     Args:
    #         matcher_model: MatcherModel
    #
    #     Returns: Nothing
    #
    #     """
    #     self._model_keys.remove(matcher_model.id)  # remove the key
    #     self.models = [ds for ds in self.models if ds.id != matcher_model.id]
    #     # remove this dataset from dataset_summary
    #     self.model_summary = self.model_summary[self.model_summary.model_id != matcher_model.id]
    #
    # def _refresh_model(self, matcher_model):
    #     """
    #     Refresh class attributes related to models.
    #     The changes affect only those models which have matcher_model.id.
    #
    #     Args:
    #         matcher_model: MatcherModel
    #
    #     Returns: Nothing.
    #
    #     """
    #     # update list of models
    #     for idx, mod in enumerate(self.models):
    #         if mod.id == matcher_model.id:
    #             if mod == matcher_model:  # no changes are needed!!!
    #                 break
    #             self.models[idx] = matcher_model
    #             # column_map should not change!
    #             break
    #
    #     # model_summary can change drastically
    #     rows = self.model_summary[self.model_summary.model_id == matcher_model.id].index.tolist()
    #     self.model_summary.set_value(rows, 'description', matcher_model.description)
    #     self.model_summary.set_value(rows, 'created', matcher_model.date_created)
    #     self.model_summary.set_value(rows, 'modified', matcher_model.date_modified)
    #     self.model_summary.set_value(rows, 'status', matcher_model.model_state.status)
    #     self.model_summary.set_value(rows, 'state_created', matcher_model.model_state.date_created)
    #     self.model_summary.set_value(rows, 'state_modified', matcher_model.model_state.date_modified)

    def __str__(self):
        return "<SchemaMatcher({})>".format(self.api)

    def __repr__(self):
        return self.__str__()

    # def _get_dataset_keys(self):
    #     """Obtain the list of keys of all available datasets."""
    #     return self._session.dataset_keys()
    #
    # def _get_model_keys(self):
    #     """Obtain the list of keys of all available models."""
    #     return self._session.model_keys()
    #
    # def _populate_datasets(self):
    #     """
    #     This method creates a dictionary (dataset key, instance of MatcherDataset).
    #     It also collects info on all datasets and puts this information
    #     into a nice Pandas data framework with the following columns:
    #     ds_key, file name, description, date created, date modified, num rows, num cols.
    #     Finally, this methods creates a dictionary to map column ids to column names.
    #
    #     Returns: dictionary of dataset ids, Pandas dataframe, dictionary of all column ids.
    #
    #     """
    #     datasets = []
    #     dataset_summary = []
    #     headers = ['dataset_id', 'name', 'description', 'created', 'modified', "num_rows", "num_cols"]
    #     column_map = dict()
    #
    #     for ds_key in self._ds_keys:
    #         dataset = api.DataSet(self._session.dataset(ds_key), self)
    #         dataset_summary.append([ds_key, dataset.filename, dataset.description,
    #                                 dataset.date_created, dataset.date_modified,
    #                                 dataset.sample.shape[0], dataset.sample.shape[1]])
    #         column_map.update(dataset._column_map)
    #         datasets.append(dataset)
    #
    #     dataset_summary = pd.DataFrame(dataset_summary)
    #     if not dataset_summary.empty: # rename columns if data frame is not empty
    #         dataset_summary.columns = headers
    #     return datasets, dataset_summary, column_map
    #
    # def _get_model_summary(self):
    #     """
    #     This method creates a dictionary (model key, instance of MatcherModel).
    #     It also collects info on all models and puts this information
    #     into a nice Pandas data framework with the following columns:
    #     model_key, description, date created, date modified, model status, its creation date and modification date.
    #
    #     Returns: dictionary of model ids, Pandas dataframe with summary of all models.
    #
    #     """
    #     df = pd.DataFrame({
    #         'model_id': [],
    #         'description': [],
    #         'created': [],
    #         'modified': [],
    #         'status': [],
    #         'state_created': [],
    #         'state_modified': []
    #     })
    #
    #     models = []
    #     for key in self._model_keys:
    #         mod = api.Model(
    #             self._session.list_model(key),
    #             self._session,
    #             column_map=self._column_map
    #         )
    #         models.append(mod)
    #
    #         # add this row into the table...
    #         df.loc[len(df)] = [
    #             key,  # model_id
    #             mod.description,  # description
    #             mod.date_created,  # created
    #             mod.date_modified,  # modified
    #             mod.model_state.status,  # status
    #             mod.model_state.date_created,  # state_created
    #             mod.model_state.date_modified  # state_modified
    #         ]
    #
    #     return models, df
    #
    # def _get_model(self, model_key):
    #     """
    #     Get the model corresponding to the key.
    #
    #     Args:
    #         model_key : key of the model at the Schema Matcher server
    #
    #     Returns: Instance of class MatcherModel corresponding to the model key
    #
    #     """
    #     return api.Model(self._session.list_model(model_key), self, self._column_map)
    #
    # def _get_dataset(self, dataset_key):
    #     """
    #     Get the dataset corresponding to the key.
    #
    #     Args:
    #         dataset_key : key of the dataset at the Schema Matcher server
    #
    #     Returns: Instance of class MatcherDataset corresponding to the model key
    #
    #     """
    #     return api.DataSet(self._session.list_dataset(dataset_key))
    #
    def create_model(self,
                     feature_config,
                     description="",
                     classes=None,
                     model_type="randomForest",
                     labels=None,
                     cost_matrix=None,
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
        if classes is None:
            classes = ["unknown"]

        def column_parse(col):
            """Turns a column into the id"""
            if issubclass(type(col), Column):
                return str(col.id)
            else:
                return str(col)

        def label_parse(ld):
            return {column_parse(k): v for k, v in ld.items()}

        assert 'unknown' in classes

        json = self.api.post_model(feature_config,
                                   description,
                                   classes,
                                   model_type,
                                   label_parse(labels),
                                   cost_matrix,
                                   resampling_strategy)  # send API request

        return NewModel(json, self)

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
        #new_dataset = #api.DataSet(self._session.post_dataset(description, file_path, type_map),
        #              #            self)
        #self._refresh_datasets()  # we need to update class attributes to include new datasets
        #
        #return new_dataset
        json = self.api.post_dataset(description, file_path, type_map)
        return NewDataSet(json, self)

    def remove_dataset(self, key):
        """
        Remove dataset `key` (can be the id, or the DataSet object)

        :param key: The key for the dataset or DataSet object
        :return: None
        """
        if issubclass(type(key), NewDataSet):
            self.api.delete_dataset(key.id)
        else:
            self.api.delete_dataset(key)

    def remove_model(self, key):
        """
        Remove model `key` (can be the id, or the Model object)

        :param key: The key for the model or Model object
        :return: None
        """
        if issubclass(type(key), NewModel):
            self.api.delete_model(key.id)
        else:
            self.api.delete_model(key)

    # def train_all(self):
    #     """
    #     Launch training for all models in the repository.
    #     Please be aware that schema matcher at the moment can handle only one model training at a time.
    #     So, running this method with wait=False might not launch training for all models.
    #
    #     Args: None
    #
    #     Returns: True if training for all models succeeded.
    #
    #     """
    #     return all([model.train() for model in self.models])
    #
    # def predict_all(self):
    #     """
    #     Launch prediction for all models for all datasets in the repository.
    #     Please be aware that schema matcher at the moment can handle only one model prediction at a time.
    #     So, running this method with wait=False is not permitted.
    #
    #     Args: None
    #
    #     Returns: Nothing, model attributes get updated.
    #
    #     """
    #     for model in self.models:
    #         model.get_predictions()


class Column(object):

    def __init__(self, json):
        """
        Initialize a Column object using a json table...

        :param json: JSON table from the server
        """
        self.index = json['index']
        self.path = json['path']
        self.name = json['name']
        self.id = json['id']
        self.size = json['size']
        self.datasetID = json['datasetID']
        self.sample = json['sample']
        self.logicalType = json['logicalType']


class NewDataSet(object):

    def __init__(self, json, parent):
        """
        Initialize a DataSet object with a json response
        :param json:
        """
        self.parent = parent
        self.id = json['id']
        self.columns = [Column(c) for c in json['columns']]
        self.filename = json['filename']
        self.path = json['path']
        self.type_map = json['typeMap']
        self.description = json['description']
        self.date_created = convert_datetime(json['dateCreated'])
        self.date_modified = convert_datetime(json['dateModified'])
        self.sample = pd.DataFrame({
            c.name: c.sample for c in self.columns
        })

    def __getitem__(self, i):
        return self.columns[i]

    def __iter__(self):
        return self.columns.__iter__()

    def __len__(self):
        return len(self.columns)

    def column(self, name):
        candidates = [c for c in self.columns if c.name == name]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise ValueError("Column name {} is ambiguous.".format(name))
        else:
            raise ValueError("Column name {} does not exist.".format(name) )

    @property
    def info(self):
        """Shows the information about the dataset"""
        return "\n".join([
            "id: {}".format(self.id),
            "columns: {}".format(self.columns),
            "filename: {}".format(self.filename),
            "path: {}".format(self.path),
            "typeMap: {}".format(self.type_map),
            "description: {}".format(self.description),
            "dateCreated: {}".format(self.date_created),
            "dateModified: {}".format(self.date_modified),
            "sample: {}".format(self.sample),
        ])


class DataSetList(collections.MutableSequence):
    """
    Container type for DataSet objects in the Serene Python Client
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not issubclass(type(v), NewDataSet):
            raise TypeError("Only DataSet types permitted: {}".format(v))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        msg = "Use SchemaMatcher.remove_dataset to correctly remove dataset"
        logging.error(msg)
        raise Exception(msg)

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __repr__(self):
        ds = []
        for v in self.list:
            s = "DataSet({})".format(v.id)
            ds.append(s)
        return "[{}]".format('\n'.join(ds))

    def to_df(self):
        df = pd.DataFrame(columns=[
            'id',
            'name',
            'description',
            'created',
            'modified',
            'rows',
            'cols'
        ])
        for elem in self.list:
            df.loc[len(df)] = [
                elem.id,
                os.path.basename(elem.filename),
                elem.description,
                elem.date_created,
                elem.date_modified,
                max(c.size for c in elem.columns),
                len(elem.columns)
            ]
        return df


class Status(Enum):
    """Enumerator of possible model states."""
    ERROR = "error"
    UNTRAINED = "untrained"
    BUSY = "busy"
    COMPLETE = "complete"

    @staticmethod
    def to_status(status):
        """Helper function to convert model state
        from a string to Status Enumerator."""
        if status == "error":
            return Status.ERROR
        if status == "untrained":
            return Status.UNTRAINED
        if status == "busy":
            return Status.BUSY
        if status == "complete":
            return Status.COMPLETE
        raise ValueError("Status {} is not supported.".format(status))


class ModelState(object):
    """
    Class to wrap the model state.

    Attributes:
        status
        message
        date_created
        date_modified
    """
    def __init__(self, json):  # status, message, date_modified):
        """
        Initialize instance of class ModelState.

        Args:
            status : string
            date_created
            date_modified
        """
        self.status = Status.to_status(json['status'])  # convert to Status enum
        self.message = json['message']
        self.date_modified = convert_datetime(json['dateChanged'])

    def __repr__(self):
        return "ModelState({}, modified on {}, msg: {})".format(
            self.status,
            self.date_modified,
            self.message
        )

    # def __eq__(self, other):
    #     if isinstance(other, ModelState):
    #         return self.status == other.status \
    #                and self.date_modified == other.date_modified
    #     return False
    #
    # def __ne__(self, other):
    #     return not self.__eq__(other)
    #
    # def __str__(self):
    #     return self.__repr__()


class NewModel(object):
    """Holds information about the Model object on the Serene server"""
    def __init__(self, json, parent):

        self._pp = pprint.PrettyPrinter(indent=4)

        self.parent = parent
        self.api = parent.api

        self.description = json['description']
        self.id = json['id']
        self.model_type = json['modelType']
        self.classes = json['classes']
        self.features = json['features']
        self.cost_matrix = json['costMatrix']
        self.resampling_strategy = json['resamplingStrategy']
        self.label_data = json['labelData']
        self.ref_datasets = json['refDataSets']
        self.model_path = json['modelPath'] if 'modelPath' in json else ''
        self.state = ModelState(json['state'])
        self.date_created = convert_datetime(json['dateCreated'])
        self.date_modified = convert_datetime(json['dateModified'])

    @property
    def info(self):
        """Shows the information about the dataset"""
        return "\n".join([
            "id: {}".format(self.id),
            "description: {}".format(self.description),
            "modelType: {}".format(self.model_type),
            "classes: {}".format('\n\t'.join(self.classes)),
            "features: {}".format(self._pp.pprint(self.features)),
            "cost_matrix: {}".format(self.cost_matrix),
            "resamplingStrategy: {}".format(self.resampling_strategy),
            "labelData: {}".format(self._pp.pprint(self.label_data)),
            "refDataSets: {}".format(self.ref_datasets),
            "modelPath: {}".format(self.model_path),
            "state: {}".format(self.state),
            "dateCreated: {}".format(self.date_created),
            "dateModified: {}".format(self.date_modified)
        ])

    @property
    def _columns(self):
        # first we grab all the columns out from the datasets
        return [ds.columns for ds in self.parent.datasets]

    @property
    def _column_lookup(self):
        # first we grab all the columns out from the datasets
        return {item.id: item for sublist in self._columns for item in sublist}

    @property
    def labels(self):
        """Returns the label DataFrame, which contains the
        user-specified columns
        """
        keys = [int(k) for k in self.label_data.keys()]
        labels = [lab for lab in self.label_data.values()]

        return pd.DataFrame({
            'user_label': labels,
            'column_name': [self._column_lookup[x].name for x in keys],
            'column_id': keys
        })

    def _label_entry(self, col, label):
        """Prepares the label entry by ensuring the key is a
           valid string key and the label is also valid"""

        if issubclass(type(col), Column):
            key = str(col.id)
        else:
            key = str(col)

        # ensure that the key is valid...
        assert \
            int(key) in self._column_lookup, \
            "Key '{}' is not in column ids {}".format(key, self._column_lookup.keys())

        # ensure that the label is in the classes...
        assert \
            label in self.classes, \
            "Label '{}' is not in classes {}".format(label, self.classes)

        return str(key), label

    def add_label(self, col, label):
        """Users can add a label to column col. `col` can be a
        Column object or a string id

        Args:
            col: Column object or int id
            label: The class label (this must exist in the model params)

        Returns:
            The updated set of labels
        """
        key, value = self._label_entry(col, label)

        label_table = self.label_data
        label_table[key] = value

        json = self.api.update_model(self.id, labels=label_table)

        self.__init__(json, self.parent)

        return self.labels

    def add_labels(self, table):
        """Users can add a label to column col. `col` can be a
        Column object or a string id

        Args:
            table: key-value dict with keys as Column objects or int ids
                   and the values as the class labels (this must exist in the model params)

        Returns:
            The updated set of labels
        """
        label_table = self.label_data

        for k, v in table.items():
            key, value = self._label_entry(k, v)
            label_table[key] = value

        json = self.api.update_model(self.id, labels=label_table)

        self.__init__(json, self.parent)

        return self.labels


class ModelList(collections.MutableSequence):
    """
    Container type for Model objects in the Serene Python Client
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not issubclass(type(v), NewModel):
            raise TypeError("Only Model types permitted: {}".format(v))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        msg = "Use SchemaMatcher.remove_model to correctly remove model"
        logging.error(msg)
        raise Exception(msg)

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __repr__(self):
        ms = []
        for v in self.list:
            s = "Model({})".format(v.id)
            ms.append(s)
        return "[{}]".format('\n'.join(ms))

    def to_df(self):
        df = pd.DataFrame(columns=[
            'model_id',
            'description',
            'created',
            'modified',
            'status',
            'state_modified'
        ])
        for elem in self.list:
            df.loc[len(df)] = [
                elem.id,
                elem.description,
                elem.date_created,
                elem.date_modified,
                elem.state.status.name,
                elem.state.date_modified
            ]
        return df


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.curdir)))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    dm = SchemaMatcher()
    print(dm.models)

    error_mods = [model for model in dm.models
                  if model.model_state.status == Status.ERROR]

    print(dm.dataset_summary)
    # print(dm.column_map)

