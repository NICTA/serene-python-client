# Python wrapper for the Schema Matcher API

# external
import logging
import requests
from urllib.parse import urljoin
import pandas as pd
from wrappers.exception_spec import BadRequestError, NotFoundError, OtherError, InternalDIError
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
    ERROR = "error"
    UNTRAINED = "untrained"
    BUSY = "busy"
    COMPLETE = "complete"

def get_status(status):
    """helper function"""
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
    """Convert string to datetime object."""
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

    def __repr__(self):
        return "<SchemaMatcherSession at (" + str(self.uri) + ")>"

    def __str__(self):
        return self.__repr__()

    def handle_errors(self, response, expr):
        """
        Raise errors based on response status_code
        Args:
            response -- response object from request
            expr -- expression where the error occurs

        :return: None or raise errors
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

        :return: list of dataset keys
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

        :return: Dictionary
        """

        logging.info('Sending request to the schema matcher server to post a dataset.')
        try:
            data = {"description": description, "file": file_path, "typeMap": type_map}
            r = self.session.post(self.uri_ds, json=data)
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
             dataset_key: integer which is the key of the dataset

        :return: dictionary
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

        :return:
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

        :return: list of model keys
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
            data["labelData"] = labels
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

        :return: model dictionary
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

        :return: model dictionary
        """

        logging.info('Sending request to the schema matcher server to update model %d' % model_key)
        uri = urljoin(self.uri_model, str(model_key))
        try:
            data = self.process_model_input(feature_config, description, classes,
                                            model_type, labels, cost_matrix, resampling_strategy)
            r = self.session.post(uri, data=data)
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

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
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

        :return: dictionary
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

        :return: True
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

        :return: True
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

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get predictions based on the model.')
        uri = urljoin(urljoin(self.uri_model, str(model_key) + "/"), "predict")
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("get_model_predict", e)
        self.handle_errors(r, "GET " + uri)
        return r.json()


class MatcherDataset(object):
    """
        Attributes:

    """
    def __init__(self, resp_dict):
        """
        Args:
            resp_dict: dictionary which is returned by Schema Matcher API
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
        self.column_map = dict() # mapping column ids to column names
        for col in self.columns:
            headers.append(col['name'])
            data.append(col['sample'])
            self.column_map[col['id']] = col['name']
        self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
        self.sample.columns = headers

    def __str__(self):
        return "<MatcherDataset(" + str(self.ds_key) + ")>"

    def __repr__(self):
        return self.__str__()


class ModelState(object):
    """
    Attributes:

    """
    def __init__(self, status, message, date_created, date_modified):
        """
        Initialize instance of class ModelState.
        Args:
            status -- string
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

    def __str__(self):
        return self.__repr__()


class MatcherModel(object):
    """
        Attributes:

    """

    def __init__(self, resp_dict, column_map=None):
        """
        Args:
            resp_dict -- dictionary which is returned by Schema Matcher API
            column_map -- optional dictionary for mapping columns fom ids to names
        """
        # TODO: check resp_dict
        try:
            self.model_key = int(resp_dict["id"])
        except Exception as e:
            logging.error("Failed to initialize MatcherModel: model key could not be converted to integer.")
            raise InternalDIError("MatcherModel initialization", e)
        self.model_type = resp_dict["modelType"]
        self.description = resp_dict["description"]
        self.features_config = resp_dict["features"]
        self.cost_matrix = resp_dict["costMatrix"]
        self.resampling_strategy = resp_dict["resamplingStrategy"]
        self.label_data = resp_dict["labelData"]
        self.ref_datasets = resp_dict["refDataSets"]
        self.classes = resp_dict["classes"]
        self.date_created = convert_datetime(resp_dict["dateCreated"])
        self.date_modified = convert_datetime(resp_dict["dateModified"])
        self.model_state = ModelState(resp_dict["state"]["status"],
                                      resp_dict["state"]["message"],
                                      resp_dict["state"]["dateCreated"],
                                      resp_dict["state"]["dateModified"])

        self.column_map = column_map

        # create dataframe with user defined labels and predictions
        self.all_data = self.get_labeldata() # column_map?
        self.features = pd.DataFrame() # empty data frame
        self.scores = pd.DataFrame() # empty data frame
        self.accuracy = np.nan # accuracy not available

    def session_update(self, api_session):
        """
        Update model from the API.
        Args:
            api_session -- schema matcher session

        :return:
        """
        cur_model = api_session.list_model(self.model_key)
        self.update(cur_model)


    def train(self, api_session, wait=True):
        """
        Send the training request to the API.
        Args:
            api_session -- schema matcher session
            wait -- boolean indicator whether to wait for the training to finish.

        :return: boolean -- True if model is trained, False otherwise
        """
        api_session.train_model(self.model_key) # launch training

        self.session_update(api_session)
        # training is done if finished
        finished = self.model_state.status == Status.ERROR or self.model_state.status == Status.COMPLETE

        while wait and not(finished):
            print("Waiting for training")
            time.sleep(30)  # wait for some time
            self.session_update(api_session)
            if self.model_state.status == Status.ERROR or\
                            self.model_state.status == Status.COMPLETE:
                finished = True # finish waiting if training failed or got complete

        return finished and self.model_state.status == Status.COMPLETE


    def get_predictions(self, api_session, wait=True):
        """
        Get predictions based on the model.
        Args:
            api_session -- schema matcher session
            wait -- boolean indicator whether to wait for the training to finish, default is True.
            column_map -- optional dictionary to lookup column names based on column ids

        :return: Pandas data framework.
        """
        train_status = self.train(api_session, wait) # do training
        api_session.predict_model(self.model_key)  # launch prediction

        self.session_update(api_session)
        # predictions are available if finished
        finished = self.model_state.status == Status.ERROR or self.model_state.status == Status.COMPLETE

        while wait and not (finished):
            time.sleep(5)  # wait for some time
            self.session_update(api_session) # update model
            if self.model_state.status == Status.ERROR or \
                            self.model_state.status == Status.COMPLETE:
                finished = True  # finish waiting if prediction failed or got complete

        if (finished and self.model_state.status == Status.COMPLETE):
            # prediction has successfully finished
            resp_dict = api_session.get_model_predict(self.model_key)
        elif self.model_state.status == Status.ERROR:
            # either training or prediction failed
            raise InternalDIError("Prediction/training failed", self.model_state.message)
        elif self.model_state.status == Status.BUSY:
            # either training or prediction is still ongoing
            raise InternalDIError("Prediction/training still in progress", self.model_state.message)
        else:
            raise InternalDIError("Training has not been launched", self.model_state.message)

        self.all_data = self.process_predictions(resp_dict)
        self.scores = self.get_scores()
        self.features = self.get_features()
        return self.all_data


    def process_predictions(self, response):
        """
        Process column predictions into Pandas data framework.
        Args:
            response-- list of column predictions (dictionaries) which is returned by the schema matcher API
            column_map -- optional dictionary to lookup column names based on column ids

        :return: Pandas data framework.
        """
        headers = ["model_id", "column_id", "column_name", "dataset_id", "actual_label", "predicted_label", "confidence"]
        classes = []
        feature_names = []
        # additionally scores for all classes and features
        data = []
        for col_pred in response:
            # if column_map is available, lookup the name of the column
            if self.column_map:
                column_name = self.column_map.get(col_pred["columnID"], np.nan)
            else:
                column_name = np.nan
            # if user provided label for this column, get the label, otherwise NaN
            actual_lab = matcher_model.label_data.get(str(col_pred["columnID"]), np.nan)
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

        all_data = pd.DataFrame(data)
        all_data.columns = headers + list(classes) + list(feature_names)
        return all_data


    def get_labeldata(self):
        headers = ["model_id", "column_id", "column_name", "actual_label"]
        # TODO: add dataset id ???
        data = []
        print("==========labeldata")
        print(self.column_map)
        print("==========labeldata")
        for columnID, label in self.label_data.items():
            if self.column_map:
                column_name = self.column_map.get(columnID, np.nan)
            else:
                column_name = np.nan
            data.append((self.model_key, columnID, column_name, label))

        sample = pd.DataFrame(data)
        sample.columns = headers
        return sample

    def get_scores(self):
        # just get a slice of all_data
        # all_data has columns:
        # ["model_id", "column_id", "column_name", "dataset_id", "actual_label", "predicted_label", "confidence"]+ classes + features
        headers = ["model_id", "column_id", "column_name", "dataset_id", "actual_label", "predicted_label",
                   "confidence"] + self.classes
        return self.all_data[headers]

    def get_features(self):
        # just get a slice of all_data
        headers = list(set(self.all_data.columns).difference(self.classes))
        return self.all_data[headers]

    def update(self, resp_dict, column_map=None):
        # TODO: check resp_dict
        try:
            self.model_key = int(resp_dict["id"])
        except Exception as e:
            logging.error("Failed to initialize MatcherModel: model key could not be converted to integer.")
            raise InternalDIError("MatcherModel initialization", e)
        self.model_type = resp_dict["modelType"]
        self.features_config = resp_dict["features"]
        self.cost_matrix = resp_dict["costMatrix"]
        self.resampling_strategy = resp_dict["resamplingStrategy"]
        self.label_data = resp_dict["labelData"]
        self.ref_datasets = set(resp_dict["refDataSets"])
        self.classes = resp_dict["classes"]
        self.date_created = convert_datetime(resp_dict["dateCreated"])
        self.date_modified = convert_datetime(resp_dict["dateModified"])
        self.model_state = ModelState(resp_dict["state"]["status"],
                                      resp_dict["state"]["message"],
                                      resp_dict["state"]["dateCreated"],
                                      resp_dict["state"]["dateModified"])
        if column_map: # update column_map if it's provided
            self.column_map = column_map

    def add_labels(self, additional_labels, api_session):
        """
        Add label data to the model.
        Args:
            additional_labels -- dictionary with additional labels for columns
            api_session -- Schema Matcher session
        """
        # labels in additional_labels extend (in case of conflict override) the ones which are currently defined
        self.label_data.update(additional_labels)
        self.session_patch(api_session, labels=self.label_data)
        self.session_update(api_session) # update model according to the changes on the server
        # reset all_data, scores and features
        self.all_data = self.get_labeldata()
        self.scores = pd.DataFrame()
        self.features = pd.DataFrame()

    def session_patch(self, api_session,
                     feature_config=None, description=None,
                     classes=None, model_type=None,
                     labels=None, cost_matrix=None, resampling_strategy=None):
        """
        Update model in the model repository at the schema matcher server.
        Args:
            api_session -- Schema Matcher session
            feature_config -- dictionary
            description -- string which describes the model to be posted
            classes -- list of class names
            model_type -- string
            labels -- dictionary
            cost_matrix -- cost matrix
            resampling_strategy -- string
        """
        # send model patch request to the schema matcher API
        api_session.update_model(self.model_key,
                     feature_config, description,
                     classes, model_type,
                     labels, cost_matrix, resampling_strategy)
        # refresh model
        self.session_update(api_session)
        # reset all_data, scores and features
        self.all_data = self.get_labeldata()
        self.scores = pd.DataFrame()
        self.features = pd.DataFrame()

    def session_delete(self, api_session):
        """
        Delete model in the model repository at the schema matcher server.
        It also destroys this instance.
        Args:
            api_session -- Schema Matcher session
        """
        api_session.delete_model(self.model_key)
        del self

    def show_info(self):
        """Construct a string which summarizes all model parameters."""
        return "model_key: " + repr(self.model_key) + "\n"\
               + "model_type: " + repr(self.model_type) + "\n"\
               + "features_config: " + repr(self.features_config) + "\n"\
               + "cost_matrix: " + repr(self.cost_matrix) + "\n"\
               + "resampling_strategy: " + repr(self.resampling_strategy) + "\n"\
               + "label_data: " + repr(self.label_data) + "\n"\
               + "ref_datasets: " + repr(self.ref_datasets) + "\n"\
               + "classes: " + repr(self.classes) + "\n"\
               + "date_created: " + str(self.date_created) + "\n"\
               + "date_modified: " + str(self.date_modified) + "\n"\
               + "model_state: " + repr(self.model_state) + "\n"

    def __str__(self):
        """Show summary of the model"""
        return self.show_info()

    def __repr__(self):
        """As indicated in the docs"""
        # show slice of all_data
        headers = list(set(["model_id", "column_id", "column_name", "dataset_id", "actual_label", "predicted_label",
                   "confidence"]).intersection(set(self.all_data.columns)))
        return repr(self.all_data[headers])

if __name__ == "__main__":

    sess = SchemaMatcherSession()
    all_ds = sess.list_alldatasets()
    all_models = sess.list_allmodels()

    model = sess.list_model(all_models[0])
    features_conf = model["features"]

    #sess.post_model(feature_config="")

    matcher_model = MatcherModel(model)

    sess.train_model(matcher_model.model_key)

    sess.predict_model(matcher_model.model_key)

    sess.get_model_predict(matcher_model.model_key)