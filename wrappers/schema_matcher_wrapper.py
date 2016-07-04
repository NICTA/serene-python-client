# Python wrapper for the Schema Matcher API

# external
import requests
import logging
import os
import datetime
from urllib.parse import urljoin
import pandas as pd

# project
from config import settings as conf_set

class SchemaMatcherSession(object):
    """
    This is the class which sets up the session for the schema matcher api.
    It provides wrappers for all calls to the API.
        Attributes:
            session: Session instance with configuration parameters for the schema matcher server from config.py
            uri: general uri for the schema matcher API
            uri_ds: uri for the dataset endpoint of the schema matcher API
            uri_model: uri for the model endpoint of the schema matcher API
    """
    def __init__(self):
        """Initialize instance of the SchemaMatcherSession."""
        logging.info('Initialising session to connect to the schema matcher server.')
        self.session = requests.Session()
        self.session.trust_env = conf_set.schema_matcher_server['trust_env']
        self.session.auth = conf_set.schema_matcher_server['auth']
        self.session.cert = conf_set.schema_matcher_server['cert']

        # uri to send requests to
        self.uri = conf_set.schema_matcher_server['uri']
        self.uri_ds = urljoin(self.uri, 'dataset') + '/' # uri for the dataset endpoint
        self.uri_model = urljoin(self.uri, 'model') + '/'  # uri for the model endpoint

    def __str__(self):
        return "<SchemaMatcherSession at (" + str(self.uri) + ")>"

    def list_alldatasets(self):
        """
        List ids of all datasets in the dataset repository at the Schema Matcher server.
        If the connection fails, empty list is returned and connection error is logged.

        :return: list of dataset keys
        """
        logging.info('Sending request to the schema matcher server to list datasets.')
        try:
            r = self.session.get(self.uri_ds)
        except Exception as e:
            logging.error(e)
            return []
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + self.uri_ds)
            return []
        return r.json()

    def post_dataset(self, description, file_path, type_map):
        """
        Post a new dataset to the schema mather server.
            Args:
                 description: string which describes the dataset to be posted
                 file_path: string which indicates the location of the dataset to be posted
                 type_map: dictionary with type map for the dataset

            :return: Dictionary
            """

        logging.info('Sending request to the schema matcher server to post a dataset.')
        try:
            data = {"description": description, "file": file_path, "typeMap": type_map}
            r = self.session.post(self.uri_ds, data = data)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during post request: status_code=' + str(r.status_code) +
                          ', uri=' + self.uri_ds)
            return
        return r.json()

    def update_dataset(self, dataset_key, description, type_map):
        """
        Update an existing dataset in the repository at the schema matcher server.
            Args:
                 description: string which describes the dataset to be posted
                 dataset_key: integer which is the dataset id
                 type_map: dictionary with type map for the dataset

            :return:
            """

        logging.info('Sending request to the schema matcher server to update dataset %d' % dataset_key)
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            data = {"description": description, "typeMap": type_map}
            r = self.session.post(uri, data=data)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during update request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
        return r.json()

    def list_dataset(self, dataset_key):
        """
        Get information on a specific dataset from the repository at the schema matcher server.
        Args:
             dataset_key: integer which is the key of the dataset

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
        return r.json()

    def delete_dataset(self, dataset_key):
        """
        Delete a specific dataset from the repository at the schema matcher server.
            Args:
                 dataset_key: int

            :return:
            """
        logging.info('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
        return r.json()

    def list_allmodels(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        :return: list of model keys
        """
        logging.info('Sending request to the schema matcher server to list models.')
        try:
            r = self.session.get(self.uri_model)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + self.uri_model)
            return
        return r.json()

    def post_model(self, description, feature_config,
                   model_type="randomForest",
                   labels=None, cost_matrix=None,
                   resampling_strategy=None):
        """
        Post a new model to the schema matcher server.
            Args:
                 description: string which describes the model to be posted
                 feature_config: dictionary
                 model_type: string
                 labels: dictionary
                 cost_matrix:
                 resampling_strategy: string

            :return: model dictionary
            """

        logging.info('Sending request to the schema matcher server to post a dataset.')
        try:
            data = {"description": description, "features": feature_config, "modelType": model_type,
                    "labelData": labels, "costMatrix": cost_matrix, "resamplingStrategy": resampling_strategy}
            r = self.session.post(self.uri_model, data=data)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during post request: status_code=' + str(r.status_code) +
                          ', uri=' + self.uri_model)
            return
        return r.json()

    def update_model(self, model_key,
                     description=None, feature_config=None,
                     model_type="randomForest",
                     labels=None, cost_matrix=None,
                     resampling_strategy=None):
        """
        Update an existing model in the model repository at the schema matcher server.
            Args:
                 model_key: integer which is the key of the model in the repository
                 description: string which describes the model to be posted
                 feature_config: dictionary
                 model_type: string
                 labels: dictionary
                 cost_matrix:
                 resampling_strategy: string

            :return: model dictionary
            """

        logging.info('Sending request to the schema matcher server to update model %d' % model_key)
        uri = urljoin(self.uri_model, str(model_key))
        try:
            data = {"description": description, "features": feature_config, "modelType": model_type,
                    "labelData": labels, "costMatrix": cost_matrix, "resamplingStrategy": resampling_strategy}
            r = self.session.post(uri, data=data)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during update request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
        return r.json()

    def list_model(self, model_key):
        """
        Get information on a specific model in the model repository at the schema matcher server.
        Args:
             model_key: integer which is the key of the model in the repository

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        uri = urljoin(self.uri_model, str(model_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
        return r.json()

    def delete_model(self, model_key):
        """
            Args:
                 model_key: integer which is the key of the model in the repository

            :return: dictionary
            """
        logging.info('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self.uri_model, str(model_key))
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
        return r.json()


class MatcherDataset(object):
    """
        Attributes:

    """
    def __init__(self, resp_dict):
        """
        Args:
            resp_dict: dictionary
        """
        self.filename = resp_dict['filename']
        self.filepath = resp_dict['path']
        self.description = resp_dict['description']
        self.ds_key = resp_dict['id']
        self.date_created = resp_dict['dateCreated']
        self.date_modified = resp_dict['dateModified']
        self.columns = resp_dict['columns']
        self.type_map = resp_dict['typeMap']

        headers = []
        data = []
        self.columnMap = []
        for col in self.columns:
            headers.append(col['name'])
            data.append(col['sample'])
            self.columnMap.append([col['id'], col['name'], col['datasetID']])
        self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
        self.sample.columns = headers

    def __str__(self):
        return "<MatcherDataset(" + str(self.ds_key) + ")>"

    def __repr__(self):
        return self.__str__()

class MatcherModel(object):
    """
        Attributes:

    """

    def __init__(self):
        """
        Args:

        """
        pass

    def __str__(self):
        return "<MatcherModel(" + str() + ")>"

    def __repr__(self):
        return self.__str__()

class SchemaMatcher(object):
    """
        Attributes:

    """

    def __init__(self):
        """
        Initialize
        """
        logging.info('Initialising schema matcher class object.')
        self.session = SchemaMatcherSession()

        self.ds_keys = self.get_dataset_keys()  # list of keys of all available datasets
        self.dataset = self.populate_datasets()

    def __str__(self):
        return "<SchemaMatcherAt(" + str(self.session.uri) + ")>"

    def __repr__(self):
        return self.__str__()

    def get_dataset_keys(self):
        """Obtain the list of keys of all available datasets."""
        return self.session.list_alldatasets()

    def populate_datasets(self):
        """
        Construct MacherDataset per each dataset key.

        :return: Dictionary for now
        """
        dataset = dict()
        for ds_key in self.ds_keys:
            dataset[ds_key] = MatcherDataset(self.session.list_dataset(ds_key))
        return dataset


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    sess = SchemaMatcherSession()
    all_ds = sess.list_alldatasets()

    dm = SchemaMatcher()

    print(dm)
    print(dm.dataset)