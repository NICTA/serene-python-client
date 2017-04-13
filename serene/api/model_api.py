import logging
from urllib.parse import urljoin

from .exceptions import InternalError
from .http import HTTPObject


class ModelAPI(HTTPObject):
    """
    Handles requests for the Model Endpoint
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI for the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'model/')

    def post(self,
             feature_config,
             description="",
             classes=None,
             model_type="randomForest",
             labels=None,
             cost_matrix=None,
             resampling_strategy="ResampleToMean",
             num_bags=50,
             bag_size=100):
        """
        Post a new model to the schema matcher server.

        :param feature_config: dictionary
        :param description: string which describes the model to be posted
        :param classes: list of class names
        :param model_type: string
        :param labels: dictionary
        :param cost_matrix:
        :param resampling_strategy: string

        :return model dictionary
        """
        if classes is None:
            classes = ["unknown"]

        assert "unknown" in classes

        logging.debug('Sending request to the schema matcher server to post a model.')
        uri = self._uri

        try:
            data = self._process_model_input(feature_config,
                                             description,
                                             classes,
                                             model_type,
                                             labels,
                                             cost_matrix,
                                             resampling_strategy,
                                             num_bags,
                                             bag_size)
            r = self.connection.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to create model", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def update(self,
               key,
               feature_config=None,
               description=None,
               classes=None,
               model_type=None,
               labels=None,
               cost_matrix=None,
               resampling_strategy=None,
               num_bags=None,
               bag_size=None):
        """
        Update an existing model in the model repository at the schema matcher server.

        :param key: integer which is the key of the model in the repository
        :param feature_config: dictionary
        :param description: string which describes the model to be posted
        :param classes: list of class names
        :param model_type: string
        :param labels: dictionary
        :param cost_matrix:
        :param resampling_strategy: string

        :return model dictionary
        """

        logging.debug('Sending request to the schema matcher server to update model %d' % key)
        uri = urljoin(self._uri, str(key))
        try:
            data = self._process_model_input(feature_config,
                                             description,
                                             classes,
                                             model_type,
                                             labels,
                                             cost_matrix,
                                             resampling_strategy,
                                             num_bags,
                                             bag_size)

            r = self.connection.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("update_model", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def item(self, key):
        """
        Get information on a specific model in the model repository at the schema matcher server.

        :param key: integer which is the key of the model in the repository

        :return: dictionary
        """
        logging.debug('Sending request to the schema matcher server to get model info.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get model", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def delete(self, key):
        """
        :param key: integer which is the key of the model in the repository

        :returns: dictionary
        """
        logging.debug('Sending request to the Serene server to delete model.')
        uri = urljoin(self._uri, str(key))

        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to delete model", e)

        self._handle_errors(r, "DELETE " + uri)
        return r.json()

    def train(self, key):
        """
        :param key: integer which is the key of the model in the repository

        :return: True, this function is asynchronous
        """
        logging.debug('Sending request to the Serene server to train the model.')
        uri = urljoin(self._uri, self.join_urls(str(key), "train"))

        try:
            r = self.connection.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to train model", e)

        self._handle_errors(r, "POST " + uri)
        return True

    def predict(self, key, dataset_key):
        """
        Post request to perform prediction based on the model, using the dataset `dataset_key`
        as input.

        Args:
            key: integer which is the key of the model in the repository
            dataset_key: integer key for the dataset to predict

        Returns: True

        """
        logging.debug('Sending request to the Serene server to preform '
                      'a prediction based on the model.')
        uri = urljoin(self._uri, self.join_urls(str(key), "predict", str(dataset_key)))

        try:
            r = self.connection.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to predict Model", e)

        self._handle_errors(r, "POST " + uri)
        logging.debug('Model prediction reply obtained.')
        return r.json()

    def keys(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        Returns: list of model keys
        """
        logging.debug('Sending request to the schema matcher server to list models.')
        uri = self._uri

        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get Model keys", e)

        self._handle_errors(r, "GET " + uri)

        return r.json()

    @staticmethod
    def _process_model_input(feature_config=None,
                             description=None,
                             classes=None,
                             model_type=None,
                             labels=None,
                             cost_matrix=None,
                             resampling_strategy=None,
                             num_bags=None,
                             bag_size=None):
        """Prepares a json update string for the model update request"""
        data = {
            "description": description,
            "classes": classes,
            "modelType": model_type,
            "labelData": {str(key): str(val) for key, val in labels.items()},
            "costMatrix": cost_matrix,
            "resamplingStrategy": resampling_strategy,
            "features": feature_config,
            "numBags": num_bags,
            "bagSize": bag_size
        }
        return {k: v for k, v in data.items() if v is not None}
