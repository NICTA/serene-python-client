"""
Python wrapper for the Serene data integration software.
"""

# external
import logging
import requests
from urllib.parse import urljoin
import pandas as pd
from serene.wrappers.exception_spec import BadRequestError, NotFoundError, OtherError, InternalDIError
from enum import Enum
from datetime import datetime
import time
import numpy as np

# project
from config import settings as conf_set

################### helper funcs

def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in
    this function because it is programmed to be pretty
    printed and may differ from the actual request.
    """
    print('{}\n{}\n{}\n\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        req.body,
    ))

class Status(Enum):
    """Enumerator of possible model states."""
    ERROR = "error"
    UNTRAINED = "untrained"
    BUSY = "busy"
    COMPLETE = "complete"

def get_status(status):
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
    raise InternalDIError("get_status("+str(status)+")", "status not supported.")

def convert_datetime(datetime_string, fmt="%Y-%m-%dT%H:%M:%SZ"):
    """
    Convert string to datetime object.

    Args:
        datetime_string: string which contains datetime object;
        fmt: format which specifies the pattern of datetime format in the string.

    Returns: datetime on successful conversion.
    Raises: InternalDIError if conversion fails.

    """
    # here we put "%Y-%m-%dT%H:%M:%SZ" to be the default format
    # T and Z are fixed literals
    # T is time separator
    # Z is literal for UTC time zone, Zulu time
    try:
        converted = datetime.strptime(datetime_string, fmt)
    except Exception as e:
        logging.error("Failed converting string to datetime: " + repr(datetime_string))
        raise InternalDIError("Failed converting string to datetime: " + repr(datetime_string), e)
    return converted
###################


class SchemaMatcherSession(object):
    """
    This is the class which sets up the session for the schema matcher api.
    It provides wrappers for all calls to the API.
    This is an internal class and should not be explicitly used/called by the user.

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
        if self.test_connection(): # test connection to the server
            logging.info("Connection to the server has been established.")
        self.uri_ds = urljoin(self.uri, 'dataset') + '/' # uri for the dataset endpoint
        self.uri_model = urljoin(self.uri, 'model') + '/'  # uri for the model endpoint

    def test_connection(self):
        """

        Returns:

        """
        logging.info('Testing connection to the server...')
        try:
            r = self.session.get(self.uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("Connection test", e)
        self.handle_errors(r, "GET " + self.uri)
        return True

    def __repr__(self):
        return "<SchemaMatcherSession at (" + str(self.uri) + ")>"

    def __str__(self):
        return self.__repr__()

    def handle_errors(self, response, expr):
        """
        Raise errors based on response status_code

        Args:
            response : response object from request
            expr : expression where the error occurs

        Returns: None or raise errors.

        Raises: BadRequestError, NotFoundError, OtherError.

        """
        if response.status_code == 200 or response.status_code == 202:
            # there are no errors here
            return

        if response.status_code == 400:
            logging.error("BadRequest in " + str(expr) + ": message='" + str(response.json()['message']) + "'")
            raise BadRequestError(expr, response.json()['message'])
        elif response.status_code == 404:
            logging.error("NotFound in " + str(expr) + ": message='" + str(response.json()['message']) + "'")
            raise NotFoundError(expr, response.json()['message'])
        else:
            logging.error("Other error with status_code=" + str(response.status_code) +
                          " in " + str(expr) + ": message='" + str(response.json()['message']) + "'")
            raise OtherError(response.status_code, expr, response.json()['message'])


    def list_alldatasets(self):
        """
        List ids of all datasets in the dataset repository at the Schema Matcher server.
        If the connection fails, empty list is returned and connection error is logged.

        Returns: list of dataset keys.
        Raises: InternalDIError on failure.
        """
        logging.info('Sending request to the schema matcher server to list datasets.')
        try:
            r = self.session.get(self.uri_ds)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_alldatasets", e)
        self.handle_errors(r, "GET " + self.uri_ds)
        return r.json()

    def post_dataset(self, description, file_path, type_map):
        """
        Post a new dataset to the schema mather server.
        Args:
             description: string which describes the dataset to be posted
             file_path: string which indicates the location of the dataset to be posted
             type_map: dictionary with type map for the dataset

        Returns: Dictionary.
        """

        logging.info('Sending request to the schema matcher server to post a dataset.')
        try:
            f = {"file": open(file_path, "rb")}
            data = {"description": str(description), "typeMap": type_map}
            r = self.session.post(self.uri_ds,  data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("post_dataset", e)
        self.handle_errors(r, "POST " + self.uri_ds)
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
            raise InternalDIError("update_dataset", e)
        self.handle_errors(r, "PATCH " + uri)
        return r.json()

    def list_dataset(self, dataset_key):
        """
        Get information on a specific dataset from the repository at the schema matcher server.
        Args:
             dataset_key: integer which is the key of the dataset.

        Returns: dictionary.
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_dataset", e)
        self.handle_errors(r, "GET " + uri)
        return r.json()

    def delete_dataset(self, dataset_key):
        """
        Delete a specific dataset from the repository at the schema matcher server.
        Args:
             dataset_key: int

        Returns:
        """
        logging.info('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("delete_dataset", e)
        self.handle_errors(r, "DELETE " + uri)
        return r.json()

    def list_allmodels(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        Returns: list of model keys
        """
        logging.info('Sending request to the schema matcher server to list models.')
        try:
            r = self.session.get(self.uri_model)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_allmodels", e)
        self.handle_errors(r, "GET " + self.uri_model)
        return r.json()

    def process_model_input(self, feature_config=None,
                   description=None,
                   classes=None,
                   model_type=None,
                   labels=None, cost_matrix=None,
                   resampling_strategy=None):

        # TODO: more sophisticated check of input...
        data = dict()
        if description:
            data["description"] = description
        if classes:
            data["classes"] = classes
        if model_type:
            data["modelType"] = model_type
        if labels:
            lab_str = dict([(str(key),str(val)) for key,val in labels.items()])
            data["labelData"] = lab_str
        if cost_matrix:
            data["costMatrix"] = cost_matrix
        if resampling_strategy:
            data["resamplingStrategy"] = resampling_strategy
        if feature_config:
            data["features"] = feature_config
        return data

    def post_model(self, feature_config,
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

        Returns: model dictionary
        """

        logging.info('Sending request to the schema matcher server to post a model.')

        try:
            data = self.process_model_input(feature_config, description, classes,
                                            model_type, labels, cost_matrix, resampling_strategy)
            r = self.session.post(self.uri_model, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("post_model", e)
        self.handle_errors(r, "POST " + self.uri_model)
        return r.json()

    def update_model(self, model_key, feature_config, description, classes,
                     model_type, labels, cost_matrix, resampling_strategy):
        """
        Update an existing model in the model repository at the schema matcher server.
        Args:
             model_key: integer which is the key of the model in the repository
             feature_config: dictionary
             description: string which describes the model to be posted
             classes: list of class names
             model_type: string
             labels: dictionary
             cost_matrix:
             resampling_strategy: string

        Returns: model dictionary
        """

        logging.info('Sending request to the schema matcher server to update model %d' % model_key)
        uri = urljoin(self.uri_model, str(model_key))
        try:
            data = self.process_model_input(feature_config, description, classes,
                                            model_type, labels, cost_matrix, resampling_strategy)
            r = self.session.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("update_model", e)
        self.handle_errors(r, "PATCH " + uri)
        return r.json()

    def list_model(self, model_key):
        """
        Get information on a specific model in the model repository at the schema matcher server.
        Args:
             model_key: integer which is the key of the model in the repository

        Returns: dictionary
        """
        logging.info('Sending request to the schema matcher server to get model info.')
        uri = urljoin(self.uri_model, str(model_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_model", e)
        self.handle_errors(r, "GET " + uri)
        return r.json()

    def delete_model(self, model_key):
        """
        Args:
             model_key: integer which is the key of the model in the repository

        Returns: dictionary
        """
        logging.info('Sending request to the schema matcher server to delete model.')
        uri = urljoin(self.uri_model, str(model_key))
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("delete_model", e)
        self.handle_errors(r, "DELETE " + uri)
        return r.json()

    def train_model(self, model_key):
        """
        Args:
             model_key: integer which is the key of the model in the repository

        Returns: True

        """
        logging.info('Sending request to the schema matcher server to train the model.')
        uri = urljoin(urljoin(self.uri_model, str(model_key)+"/"), "train")
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("train_model", e)
        self.handle_errors(r, "GET " + uri)
        return True

    def predict_model(self, model_key):
        """
        Post request to perform prediction based on the model.

        Args:
             model_key: integer which is the key of the model in the repository

        Returns: True

        """
        logging.info('Sending request to the schema matcher server to preform prediction based on the model.')
        uri = urljoin(urljoin(self.uri_model, str(model_key) + "/"), "predict")
        try:
            r = self.session.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("predict_model", e)
        self.handle_errors(r, "POST " + uri)
        return True

    def get_model_predict(self, model_key):
        """
        Get predictions based on the model.

        Args:
             model_key: integer which is the key of the model in the repository

        Returns: dictionary

        """
        logging.info('Sending request to the schema matcher server to get predictions based on the model.')
        uri = urljoin(urljoin(self.uri_model, str(model_key) + "/"), "predict")
        try:
            resp = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("get_model_predict", e)
        self.handle_errors(resp, "GET " + uri)
        return resp.json()
######################################


class MatcherDataset(object):
    """
        Attributes:
            id: dataset id by which it is referred on the server
            filename: name of the file
            filepath: string which indicates the location of the file on the server
            description: description of the file
            date_created: date when the file was created
            date_modified: last date when the dataset was modified
            type_map: optional type map
            sample: Pandas dataframe with data sample

            _matcher
            _column_map
    """
    def __init__(self, resp_dict, matcher):
        """
        Initialize instance of class MatcherDataset.

        Args:
            resp_dict: dictionary which is returned by Schema Matcher API
            matcher: instance of class SchemaMatcher, it contains session as attribute

        """
        try:
            self.id = int(resp_dict["id"])
        except Exception as e:
            logging.error("Failed to initialize MatcherDataset: dataset key could not be converted to integer.")
            raise InternalDIError("MatcherDataset initialization", e)

        self.filename = resp_dict['filename']
        self.filepath = resp_dict['path']
        self.description = resp_dict['description']
        self.date_created = resp_dict['dateCreated']
        self.date_modified = resp_dict['dateModified']
        self.type_map = resp_dict['typeMap']

        self._matcher = matcher

        headers = []
        data = []
        self._column_map = dict() # mapping column ids to column names
        for col in resp_dict['columns']:
            headers.append(col['name'])
            data.append(col['sample'])
            self._column_map[int(col['id'])] = col['name']
        self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
        self.sample.columns = headers


    def __eq__(self,other):
        if isinstance(other, MatcherDataset):
            return (self.id == other.id) and (self.filepath == other.filepath) \
                   and (self.filename == other.filename) and (self.description == other.description) \
                   and (self.date_modified == other.date_modified) \
                   and (self.date_created == other.date_created) and (self.sample.equals(other.sample)) \
                   and (self._column_map == other._column_map)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _refresh(self, resp_dict):
        """
        Refresh instance attributes.

        Args:
            resp_dict: dictionary which is returned by Schema Matcher API

        Returns: Nothing.
        """
        try:
            self.id = int(resp_dict["id"])
        except Exception as e:
            logging.error("Failed to initialize MatcherDataset: dataset key could not be converted to integer.")
            raise InternalDIError("MatcherDataset initialization", e)

        self.filename = resp_dict['filename']
        self.filepath = resp_dict['path']
        self.description = resp_dict['description']
        self.date_created = resp_dict['dateCreated']
        self.date_modified = resp_dict['dateModified']
        self.type_map = resp_dict['typeMap']

        headers = []
        data = []
        self._column_map = dict() # mapping column ids to column names
        for col in resp_dict['columns']:
            headers.append(col['name'])
            data.append(col['sample'])
            self._column_map[int(col['id'])] = col['name']
        self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
        self.sample.columns = headers


    def __str__(self):
        return "<MatcherDataset(" + str(self.id) + ")>"

    def __repr__(self):
        return self.__str__()

    def delete(self):
        """
        Delete dataset in the dataset repository at the schema matcher server.
        This method also updates attributes in the matcher instance.
        It also destroys this instance.

        Returns: Nothing.

        """
        logging.info("Deleting MatcherDataset %d" % self.id)
        self._matcher._session.delete_dataset(self.id)
        # delete references in self._matcher
        # self._matcher._ds_keys
        # self._matcher.datasets
        # self._matcher.dataset_summary
        # self._matcher._column_map
        self._matcher._remove_dataset(self)

        del self

    def update(self, description=None, type_map=None):
        """
        Update the dataset with new description/type_map on the server.
        If any of the parameters are empty, the current values of the dataset are used.

        Args:
            description: new description for the dataset
            type_map: new type_map for the dataset

        Returns: Nothing.

        """
        if description is None:
            description = self.description
        if type_map is None:
            type_map = self.type_map
        new_dict = self._matcher._session.update_dataset(self.id, description, type_map)
        self._refresh(new_dict)
        self._matcher._refresh_dataset(self)

    def construct_labelData(self, filepath):
        """
        We want to construct a dictionary {column_id:class_label} for the dataset based on a .csv file.
        This method reads in .csv as a Pandas data frame, selects columns "column_name" and "class",
        drops NaN and coverts these two columns into dictionary.
        We obtain a lookup dictionary
        where the key is the column name and the value is the class label.
        Then by using column_map the method builds the required dictionary.

        Args:
            filepath: string where .csv file is located.

        Returns: dictionary

        """
        label_data = {}  # we need this dictionary (column_id, class_label)
        try:
            frame = pd.read_csv(filepath)
            # dictionary (column_name, class_label)
            name_labels = frame[["column_name", "class"]].dropna().set_index('column_name')['class'].to_dict()
            for col_id, col_name in self._column_map.items():
                if col_name in name_labels:
                    label_data[int(col_id)] = name_labels[col_name]
        except Exception as e:
            raise InternalDIError("construct_labelData", e)

        return label_data
######################################


class ModelState(object):
    """
    Class to wrap the model state.

    Attributes:
        status
        message
        date_created
        date_modified
    """
    def __init__(self, status, message, date_created, date_modified):
        """
        Initialize instance of class ModelState.

        Args:
            status : string
            date_created
            date_modified
        """
        self.status = get_status(status) # convert to Status enum
        self.message = message
        self.date_created = convert_datetime(date_created)
        self.date_modified = convert_datetime(date_modified)

    def __repr__(self):
        return "ModelState(" + repr(self.status)\
               + ", created on " + str(self.date_created)\
               + ", modified on " + str(self.date_modified)\
               + ", message " + repr(self.message) + ")"

    def __eq__(self, other):
        if isinstance(other, ModelState):
            return self.status == other.status and self.date_created == other.date_created \
                   and self.date_modified == other.date_modified
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.__repr__()


class MatcherModel(object):
    """
    Class to wrap the model.

    Attributes:
        id: model key by which the model is refered on the server
        model_type: type of the model
        description: description of the model
        features_config: model configuration of the features
        cost_matrix: cost matrix
        resampling_strategy: resampling strategy of the model
        ref_datasets: list of dataset ids which are referenced by labeled data
        classes: list of semantic types
        date_created: date when model was created
        date_modified: last date when model was modified
        model_state: information about model state

        _label_data: internal
        _matcher: internal
        _column_map: internal

        all_data: Pandas dataframe
        features: Pandas dataframe
        scores: Pandas dataframe
        user_labels: Pandas dataframe

    """

    def __init__(self, resp_dict, matcher, column_map=None):
        """
        Args:
            resp_dict : dictionary which is returned by Schema Matcher API
            matcher: instance of class SchemaMatcher, it contains session as attribute
            column_map : optional dictionary for mapping columns fom ids to names
        """
        # TODO: check resp_dict
        try:
            self.id = int(resp_dict["id"])
        except Exception as e:
            logging.error("Failed to initialize MatcherModel: model key could not be converted to integer.")
            raise InternalDIError("MatcherModel initialization", e)
        self.model_type = resp_dict["modelType"]
        self.description = resp_dict["description"]
        self.features_config = resp_dict["features"]
        self.cost_matrix = resp_dict["costMatrix"]
        self.resampling_strategy = resp_dict["resamplingStrategy"]
        self._label_data = {} # this attribute is internal and should not be viewed/modified by the user
        try:
            # print(resp_dict["labelData"])
            # print(type(resp_dict["labelData"]))
            for col_id, lab in resp_dict["labelData"].items():
                self._label_data[int(col_id)] = lab
        except Exception as e:
            raise InternalDIError("failed to convert labelData",e)
        self.ref_datasets = set(resp_dict["refDataSets"])
        self.classes = resp_dict["classes"]
        self.date_created = convert_datetime(resp_dict["dateCreated"])
        self.date_modified = convert_datetime(resp_dict["dateModified"])
        self.model_state = ModelState(resp_dict["state"]["status"],
                                      resp_dict["state"]["message"],
                                      resp_dict["state"]["dateCreated"],
                                      resp_dict["state"]["dateModified"])


        self._matcher = matcher  # this attribute is internal and should not be viewed/modified by the user
        self._column_map = column_map  # this attribute is internal and should not be viewed/modified by the user

        # create dataframe with user defined labels and predictions
        self.all_data = self._get_labeldata() # column_map?
        self.features = pd.DataFrame() # empty data frame
        self.scores = pd.DataFrame() # empty data frame
        self.user_labels = self.all_data.copy()

    def __eq__(self, other):
        if isinstance(other, MatcherModel):
            return (self.id == other.id) and (self.model_type == other.model_type) \
                   and (self.model_state == other.model_state) and (self.date_created == other.date_created) \
                   and (self.date_modified == other.date_modified) and (self.classes == other.classes) \
                   and (self.ref_datasets == other.ref_datasets) \
                   and (self.features_config == other.features_config) and (self.description == other.description) \
                   and (self.all_data.equals(other.all_data)) and (self.scores.equals(other.scores)) \
                   and (self.features.equals(other.features)) and (self.user_labels.equals(other.user_labels))
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


    def _session_update(self):
        """
        Download model updates from the API.
        """
        cur_model = self._matcher._session.list_model(self.id)
        self._refresh(cur_model)


    def train(self, wait=True):
        """
        Send the training request to the API.

        Args:
            wait : boolean indicator whether to wait for the training to finish.

        Returns: boolean -- True if model is trained, False otherwise

        """
        self._matcher._session.train_model(self.id) # launch training

        self._session_update()
        # training is done if finished
        finished = self.model_state.status == Status.COMPLETE\
                   or self.model_state.status == Status.ERROR

        while wait and not finished: # TODO: change to thread blocking
            logging.info("Waiting for the training...")
            time.sleep(5)  # wait for some time
            self._session_update()
            if self.model_state.status == Status.ERROR or\
                            self.model_state.status == Status.COMPLETE:
                finished = True # finish waiting if training failed or got complete

        return finished and self.model_state.status == Status.COMPLETE


    def get_predictions(self, wait=True, dataset_id=None): # TODO: additional parameter for dataset_id
        """
        Get predictions based on the model.
        Predictions are done for all datasets in the repository if dataset_id=None.
        If a dataset id is passed to the method, prediction will be done only for this dataset.

        Args:
            wait : boolean indicator whether to wait for the training to finish, default is True.
            dataset_id: dataset id for which the prediction should be done;
                        if None, prediction is done for all datasets (this is default).

        Returns: Pandas data framework.

        """
        train_status = self.train(wait) # do training
        self._matcher._session.predict_model(self.id)  # launch prediction

        self._session_update()
        # predictions are available if finished
        finished = self.model_state.status == Status.COMPLETE\
                   or self.model_state.status == Status.ERROR

        while wait and not finished:
            time.sleep(3)  # wait for some time
            # The sleep function is an OS call that differs from busy wait in that it doesn't block the thread.
            # TODO: should we switch to threading.Event?
            self._session_update() # update model
            if self.model_state.status == Status.ERROR or \
                            self.model_state.status == Status.COMPLETE:
                finished = True  # finish waiting if prediction failed or got complete

        if finished and self.model_state.status == Status.COMPLETE:
            # prediction has successfully finished
            resp_dict = self._matcher._session.get_model_predict(self.id)
        elif self.model_state.status == Status.ERROR:
            # either training or prediction failed
            raise InternalDIError("Prediction/training failed", self.model_state.message)
        elif self.model_state.status == Status.BUSY:
            # either training or prediction is still ongoing
            raise InternalDIError("Prediction/training still in progress", self.model_state.message)
        else:
            raise InternalDIError("Training has not been launched", self.model_state.message)

        self.all_data = self._process_predictions(resp_dict)
        self.scores = self._get_scores()
        self.features = self._get_features()
        return self.all_data

    def _process_predictions(self, response):
        """
        Process column predictions into Pandas data framework.

        Args:
            response: list of column predictions (dictionaries) which is returned by the schema matcher API

        Returns: Pandas data framework.

        """
        headers = ["model_id", "column_id", "column_name", "dataset_id", "user_label", "predicted_label", "confidence"]
        classes = []
        feature_names = []
        # additionally scores for all classes and features
        data = []

        for col_pred in response:
            # if column_map is available, lookup the name of the column
            if self._column_map:
                column_name = self._column_map.get(col_pred["columnID"], np.nan)
            else:
                column_name = np.nan
            # if user provided label for this column, get the label, otherwise NaN
            actual_lab = self._label_data.get(str(col_pred["columnID"]), np.nan)
            conf = col_pred["confidence"]
            if conf > 0:
                label = col_pred["label"]
            else:
                label = np.nan # training actually failed...
            new_row = (col_pred["id"], col_pred["columnID"], column_name, col_pred["datasetID"],
                       actual_lab, label, conf)
            if len(classes) == 0: # get names of classes
                classes = col_pred["scores"].keys()
            if len(feature_names) == 0: # get names of features
                feature_names = col_pred["features"].keys()

            new_row += tuple(col_pred["scores"].values()) # get scores for all classes
            new_row += tuple(col_pred["features"].values())  # get features which were used for prediction
            data.append(new_row)

        if not len(data):
            logging.warning("Predictions are not available!")
            return self.all_data
        all_data = pd.DataFrame(data)
        # TODO: what should be done if the data is empty?
        all_data.columns = headers + list(classes) + list(feature_names)
        return all_data

    def calculate_confusionMatrix(self):
        """
        Calculate confusion matrix of the model.
        If all_data is not available for the model, it will return None.

        Returns: Pandas data frame.

        """
        if self.all_data.empty:
            logging.warning("Model all_data is empty. Confusion matrix cannot be calculated.")
            return None

        # take those rows where user_label is available
        try:
            available = self.all_data[["user_label", "predicted_label"]].dropna()
        except Exception as e:
            raise InternalDIError("confusion_matrix",e)
        y_actu = pd.Series(available["user_label"], name='Actual')
        y_pred = pd.Series(available["predicted_label"], name='Predicted')
        return pd.crosstab(y_actu, y_pred)

    def _get_labeldata(self):
        """
        Creates a Pandas dataframe which contains user specified labels for columns.

        Returns: Pandas dataframe with columns
                "model_id", "column_id", "column_name", "user_label".

        """
        headers = ["model_id", "column_id", "column_name", "user_label"]
        # TODO: add dataset id ???
        data = []
        for columnID, label in self._label_data.items():
            if self._column_map:
                column_name = self._column_map.get(columnID, np.nan)
            else:
                column_name = np.nan
            data.append((self.id, columnID, column_name, label))

        sample = pd.DataFrame(data).dropna()
        sample.columns = headers
        return sample

    def _get_scores(self):
        """
        Get a slice of all_data which has scores for classes.

        Returns: Pandas data framework

        """
        # just get a slice of all_data
        # all_data has columns:
        # ["model_id", "column_id", "column_name", "dataset_id", "user_label", "predicted_label", "confidence"]+ classes + features
        headers = ["model_id", "column_id", "column_name", "dataset_id", "user_label", "predicted_label",
                   "confidence"] + self.classes
        if len(set(headers).intersection(set(self.all_data.columns))) == len(headers):
            return self.all_data[headers]
        logging.warning("Scores are not available for model: " + str(self.id))
        return pd.DataFrame()

    def _get_features(self):
        """
        Get a slice of all_data which has features calculated for columns in the dataset repo.

        Returns: Pandas data framework

        """
        # just get a slice of all_data
        headers = list(set(self.all_data.columns).difference(self.classes))
        return self.all_data[headers]

    def _refresh(self, resp_dict, column_map=None):
        """
        Update the model according to the newly provided parameters.

        Args:
            resp_dict: dictionary with new parameters
            column_map: dictionary of column ids

        Returns: Nothing.

        """
        # TODO: check resp_dict
        try:
            self.id = int(resp_dict["id"])
        except Exception as e:
            logging.error("Failed to initialize MatcherModel: model key could not be converted to integer.")
            raise InternalDIError("MatcherModel initialization", e)
        self.model_type = resp_dict["modelType"]
        self.features_config = resp_dict["features"]
        self.cost_matrix = resp_dict["costMatrix"]
        self.description = resp_dict["description"]
        self.resampling_strategy = resp_dict["resamplingStrategy"]
        self._label_data = resp_dict["labelData"]
        self.ref_datasets = set(resp_dict["refDataSets"])
        self.classes = resp_dict["classes"]
        self.date_created = convert_datetime(resp_dict["dateCreated"])
        self.date_modified = convert_datetime(resp_dict["dateModified"])
        self.model_state = ModelState(resp_dict["state"]["status"],
                                      resp_dict["state"]["message"],
                                      resp_dict["state"]["dateCreated"],
                                      resp_dict["state"]["dateModified"])
        if column_map: # update column_map if it's provided
            self._column_map = column_map

        # refresh _matcher
        self._matcher._refresh_model(self)

        # reset all_data, scores and features
        self.all_data = self._get_labeldata()
        self.scores = pd.DataFrame()
        self.features = pd.DataFrame()
        self.user_labels = self.all_data.copy()

    def add_labels(self, additional_labels):
        """
        Add label data to the model and upload the changes to the server.

        Args:
            additional_labels : dictionary with additional labels for columns

        Returns: Nothing.

        """
        # labels in additional_labels extend (in case of conflict override) the ones which are currently defined
        self._label_data.update(additional_labels)
        self.update(labels=self._label_data)

        # reset all_data, scores and features --- this is already done in update method
        # self.all_data = self._get_labeldata()
        # self.user_labels = self.all_data.copy()
        # self.scores = pd.DataFrame()
        # self.features = pd.DataFrame()

    def update(self,
               feature_config=None, description=None,
               classes=None, model_type=None,
               labels=None, cost_matrix=None, resampling_strategy=None):
        """
        Update model in the model repository at the schema matcher server.
        If any of the parameters is None, this method will take the current model value for it.

        Args:
            feature_config : dictionary
            description : string which describes the model to be posted
            classes : list of class names
            model_type : string
            labels : dictionary
            cost_matrix : cost matrix
            resampling_strategy : string

        Returns:
        """
        # if any of the parameters is None, then take the current one
        if feature_config is None:
            feature_config = self.features_config
        if description is None:
            description = self.description
        if classes is None:
            classes = self.classes
        if model_type is None:
            model_type = self.model_type
        if labels is None:
            labels = self._label_data
        if cost_matrix is None:
            cost_matrix = self.cost_matrix
        if resampling_strategy is None:
            resampling_strategy = self.resampling_strategy
        # send model patch request to the schema matcher API
        new_dict = self._matcher._session.update_model(self.id,
                                 feature_config, description,
                                 classes, model_type,
                                 labels, cost_matrix, resampling_strategy)
        # refresh model
        self._refresh(new_dict)

    def delete(self):
        """
        Delete model in the model repository at the schema matcher server.
        This method also updates attributes in the matcher instance.
        It also destroys this instance.
        """
        logging.info("Deleting MatcherModel %d" % self.id)
        self._matcher._session.delete_model(self.id)
        # we need to delete reference to the model in self._matcher
        # self._matcher.models
        # self._matcher.model_summary
        # self._matcher.model_keys
        self._matcher._remove_model(self)

        del self

    def show_info(self):
        """Construct a string which summarizes all model parameters."""
        return "model_key: " + repr(self.id) + "\n"\
               + "model_type: " + repr(self.model_type) + "\n"\
               + "features_config: " + repr(self.features_config) + "\n"\
               + "cost_matrix: " + repr(self.cost_matrix) + "\n"\
               + "resampling_strategy: " + repr(self.resampling_strategy) + "\n"\
               + "label_data: " + repr(self._label_data) + "\n"\
               + "ref_datasets: " + repr(self.ref_datasets) + "\n"\
               + "classes: " + repr(self.classes) + "\n"\
               + "date_created: " + str(self.date_created) + "\n"\
               + "date_modified: " + str(self.date_modified) + "\n"\
               + "model_state: " + repr(self.model_state) + "\n"

    def __str__(self):
        """Show summary of the model"""
        return self.show_info()

    def __repr__(self):
        """Different from the original docs"""
        # show slice of all_data
        # headers = list(set(["model_id", "column_id", "column_name", "dataset_id", "user_label", "predicted_label",
        #            "confidence"]).intersection(set(self.all_data.columns)))
        return "<MatcherModel(" + str(self.id) + ")>"
        # return repr(self.all_data[headers]) # according to the docs

if __name__ == "__main__":

    sess = SchemaMatcherSession()
    all_ds = sess.list_alldatasets()
    all_models = sess.list_allmodels()
