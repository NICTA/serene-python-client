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
import time

from .utils import convert_datetime
from .wrappers import schema_matcher as api
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

    @staticmethod
    def confusion_matrix(prediction):
        """
        Calculate confusion matrix of the model.
        If all_data is not available for the model, it will return None.

        Args:
            prediction: The prediction dataframe, must have label and user_label

        Returns: Pandas data frame.

        """
        if prediction.empty:
            logging.warning("Model all_data is empty. Confusion matrix cannot be calculated.")
            return None

        # take those rows where user_label is available
        available = prediction[["user_label", "label"]].dropna()

        y_actu = pd.Series(available["user_label"], name='Actual')
        y_pred = pd.Series(available["label"], name='Predicted')
        return pd.crosstab(y_actu, y_pred)

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

    def train_all(self):
        """
        Launch training for all models in the repository.

        Args: None

        Returns: True if training for all models succeeded.

        """
        return all([model.train() for model in self.models])

    def predict_all(self, model, scores=True, features=False):
        """
        Launch prediction for all models for all datasets in the repository.
        Please be aware that schema matcher at the moment can handle only one model prediction at a time.
        So, running this method with wait=False is not permitted.

        Args: None

        Returns: Nothing, model attributes get updated.

        """
        results = [model.predict(d, scores, features) for d in self.datasets]
        return pd.concat(results)

    def __repr__(self):
        return "<SchemaMatcher({})>".format(self.api)


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
    def summary(self):
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

    @property
    def summary(self):
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
        if len(self.message):
            return "ModelState({}, modified on {}, msg: {})".format(
                self.status,
                self.date_modified,
                self.message
            )
        else:
            return "ModelState({}, modified on {})".format(
                self.status,
                self.date_modified
            )


class NewModel(object):
    """Holds information about the Model object on the Serene server"""
    def __init__(self, json, parent):

        self._pp = pprint.PrettyPrinter(indent=4)

        self.parent = parent
        self.api = parent.api
        self.PREDICT_KEYS = [
            "column_id",
            "confidence",
            "dataset_id",
            "model_id",
            "label",
            "user_label"
        ]
        self.PREDICT_SCORE_PRE = "scores"
        self.PREDICT_FEATURE_PRE = "features"

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

        self._update(json)

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

        self._update(json)

        return self.labels

    def train(self):
        """
        Send the training request to the API.

        Args:
            wait : boolean indicator whether to wait for the training to finish.

        Returns: boolean -- True if model is trained, False otherwise

        """
        self.api.train_model(self.id)  # launch training

        def state():
            """Query the server for the model state"""
            json = self.api.model(self.id)
            self._update(json)
            return self.state

        def is_finished():
            """Check if training is finished"""
            return state().status in {Status.COMPLETE, Status.ERROR}

        print("Training model {}...".format(self.id))
        while not is_finished():
            logging.info("Waiting for the training to complete...")
            time.sleep(2)  # wait in polling loop

        print("Training complete for {}".format(self.id))
        logging.info("Training complete for {}.".format(self.id))
        return state().status == Status.COMPLETE

    def predict(self, dataset, scores=True, features=False):
        """Runs a prediction across the `dataset`"""
        df = self._full_predict(dataset)

        keys = [k for k in self.PREDICT_KEYS]

        if features:
            keys += [col for col in df.columns if self.PREDICT_FEATURE_PRE in col]

        if scores:
            keys += [col for col in df.columns if self.PREDICT_SCORE_PRE in col]

        return df[keys]

    @property
    def is_error(self):
        """Returns True if model is in the error state"""
        return self.state.status == Status.ERROR

    @property
    def summary(self):
        """Shows the information about the dataset"""
        df = {
            "id": self.id,
            "description": self.description,
            "modelType": self.model_type,
            "classes": self.classes,
            "features": self.features,
            "cost_matrix": self.cost_matrix,
            "resamplingStrategy": self.resampling_strategy,
            "labelData": self.label_data,
            "refDataSets": self.ref_datasets,
            "modelPath": self.model_path,
            "state": self.state,
            "dateCreated": self.date_created,
            "dateModified": self.date_modified
        }
        str = self._pp.pformat(df)
        return str

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

    @property
    def all_data(self):
        return pd.DataFrame()

    def _full_predict(self, dataset):
        """
        Predict the column labels for this dataset

        :param dataset: Can be dataset id or a DataSet object

        :return: Pandas Dataframe with prediction data
        """
        if issubclass(type(dataset), NewDataSet):
            key = dataset.id
        else:
            key = int(dataset)

        json = self.api.predict_model(self.id, key)

        df = self._predictions(json)

        return df

    def _update(self, json):
        """Re-initializes the model based on an updated json string"""
        self.__init__(json, self.parent)

    def _columns(self):
        # first we grab all the columns out from the datasets
        return [ds.columns for ds in self.parent.datasets]

    @property
    def _column_lookup(self):
        # first we grab all the columns out from the datasets
        return {item.id: item for sublist in self._columns() for item in sublist}

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

    def _predictions(self, json):
        """
        Here we flatten out the nested json returned from Serene
        into a flat table structure.

        Initially we have:
            datasetID: {
                item1 : asdf
                item2 : qwer
                nested-item1: {
                    item1: asdf
                    item2: qwer
                }
                nested-item2: {
                    item1: asdf
                    item2: qwer
                }
            }

        and we want to flatten it to a DataFrame with:

            'datasetID': [...]
            'item1': [...]
            'item2': [...]
            'nested-item1-item1: [...]
            'nested-item1-item2: [...]
            'nested-item2-item1: [...]
            'nested-item2-item2: [...]

        :param json: JSON returned from the backend
        :return: Flattened dictionary of lists for each key
        """
        table = self._flat_predict(json)

        df = pd.DataFrame(table)

        # now we add the user labels
        return pd.merge(
            df,
            self.labels[['column_id', 'user_label']],
            on='column_id',
            how='left'
        )

    def _flat_predict(self, json):
        """
         Here we flatten out the nested json returned from Serene
        into a flat table structure.

        Initially we have:
            datasetID: {
                item1 : asdf
                item2 : qwer
                nested-item1: {
                    item1: asdf
                    item2: qwer
                }
                nested-item2: {
                    item1: asdf
                    item2: qwer
                }
            }

        and we want to flatten it to a DataFrame with:
            {
                'datasetID': [...]
                'item1': [...]
                'item2': [...]
                'nested-item1-item1: [...]
                'nested-item1-item2: [...]
                'nested-item2-item1: [...]
                'nested-item2-item2: [...]
            }

        :param json: JSON returned from the backend
        :return: Flattened dictionary of lists for each key

        """
        # first we need to drop the datasetID key down
        def update(d, key, val):
            """Better dictionary update function"""
            d[key] = val
            return d

        # first store the IDs for later...
        datasetID = json['dataSetID']
        modelID = json['modelID']

        # next drop the key into a "columnID":key inside the child dict
        dlist = [update(d, "column_id", int(k)) for k, d in json['predictions'].items()]

        # we now want to flatten the nested items
        flat_list = [self._flatten(d) for d in dlist]

        # now we add the IDs as well
        final = [update(d, "dataset_id", int(datasetID)) for d in flat_list]
        final = [update(d, "model_id", int(modelID)) for d in final]

        table = collections.defaultdict(list)

        # WARNING: This is dangerous! If a value is missing it will
        # break! We need to add np.nan as missing values if there
        # are missing elements...
        for d in final:
            for k, v in d.items():
                table[k].append(v)
        return table

    def _flatten(self, d, parent_key='', sep='_'):
        """
        Flattens a nested dictionary by squashing the
        parent keys into the sub-key name with separator
        e.g.
            flatten({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
            >> {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}

        :param d: The nested dictionary
        :param parent_key: parent key prefix
        :param sep: The separator to
        :return:
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


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

    @property
    def summary(self):
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

