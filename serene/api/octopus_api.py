import logging
from urllib.parse import urljoin
import json

from .exceptions import InternalError
from .http import HTTPObject


class OctopusAPI(HTTPObject):
    """
    Handles requests for the Octopus Endpoint
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI for the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'octopus/')

    def post(self,
             ssds=None,
             name="",
             description="",
             feature_config=None,
             model_type="randomForest",
             resampling_strategy="NoResampling",
             num_bags=10,
             bag_size=10,
             ontologies=None,
             modeling_props=None):
        """
        Post a new model to the schema matcher server.

        :param ssds: List of ids for the semantic source descriptions which will form the training dataset
        :param name: user-defined name for this octopus
        :param description: string which describes the model to be posted
        :param feature_config: dictionary
        :param model_type: string
        :param resampling_strategy: string
        :param num_bags: number of bags to be created per column
        :param bag_size: number of rows to be sampled from the column per each bag
        :param ontologies: List of ids for owls
        :param modeling_props: configuration parameters for semantic modeler
        :return octopus dictionary
        """
        logging.debug('Sending request to the serene server to post octopus.')
        uri = self._uri

        try:
            data = self._process_octopus_input(ssds,
                                               name,
                                               description,
                                               feature_config,
                                               model_type,
                                               resampling_strategy,
                                               num_bags,
                                               bag_size,
                                               ontologies,
                                               modeling_props)
            r = self.connection.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to create octopus ", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def update(self,
               key,
               ssds=None,
               name=None,
               description=None,
               feature_config=None,
               model_type=None,
               resampling_strategy=None,
               num_bags=None,
               bag_size=None,
               ontologies=None,
               modeling_props=None):
        """
        Update an existing octopus in the repository at the serene server.

        :param key: integer which is the key of the octopus in the repository
        :param ssds: List of ids for the semantic source descriptions which will form the training dataset
        :param name: user-defined name for this octopus
        :param description: string which describes the model to be posted
        :param feature_config: dictionary
        :param model_type: string
        :param resampling_strategy: string
        :param num_bags: number of bags to be created per column
        :param bag_size: number of rows to be sampled from the column per each bag
        :param ontologies: List of ids for owls
        :param modeling_props: configuration parameters for semantic modeler

        :return model dictionary
        """

        logging.debug('Sending request to the serene server to update octopus %d' % key)
        uri = urljoin(self._uri, str(key))
        try:
            data = self._process_octopus_input(ssds, name, description, feature_config, model_type,
                                               resampling_strategy, num_bags, bag_size, ontologies,
                                               modeling_props)
            r = self.connection.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("Update octopus ", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def item(self, key):
        """
        Get information on a specific octopus in the repository at the serene server.

        :param key: integer which is the key of the octopus in the repository

        :return: dictionary
        """
        logging.debug('Sending request to the serene server to get octopus %d info' % key)
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get octopus ", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def alignment(self, key):
        """
        Get alignmnt graph for a specific octopus in the repository at the serene server.

        :param key: integer which is the key of the octopus in the repository

        :return: dictionary
        """
        logging.debug('Sending request to the serene server to get alignment graph for the octopus %d' % key)
        uri = urljoin(self._uri, str(key)+"/")
        uri = urljoin(uri, "alignment")
        try:
            print("alignment uri: ", uri)
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get octopus ", e)
        self._handle_errors(r, "GET " + uri)
        # server returns a string which we load with json
        return json.loads(r.json())

    def delete(self, key):
        """
        :param key: integer which is the key of the octopus in the repository

        :returns: dictionary
        """
        logging.debug('Sending request to the Serene server to delete octopus %d.' % key)
        uri = urljoin(self._uri, str(key))

        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to delete octopus ", e)

        self._handle_errors(r, "DELETE " + uri)
        return r.json()

    def train(self, key):
        """
        :param key: integer which is the key of the octopus in the repository

        :return: True, this function is asynchronous
        """
        logging.debug('Sending request to the Serene server to train the octopus %d' % key)
        # lobster (schema matcher model) training will be launched automatically on the server side
        uri = urljoin(self._uri, self.join_urls(str(key), "train"))

        try:
            r = self.connection.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to train octopus ", e)

        self._handle_errors(r, "POST " + uri)
        return True

    def predict(self, key, dataset_key):
        """
        Post request to perform prediction based on the octopus, using the dataset `dataset_key`
        as input.

        Args:
            key: integer which is the key of the octopus in the repository
            dataset_key: integer key for the dataset to predict

        Returns: JSON object with predicted semantic models and associated scores in there

        """
        logging.debug('Sending request to the Serene server to perform '
                      'a prediction based on the octopus {} for the dataset {}'.format(key, dataset_key))
        uri = urljoin(self._uri, self.join_urls(str(key), "predict", str(dataset_key)))

        try:
            r = self.connection.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to predict octopus ", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def keys(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        Returns: list of model keys
        """
        logging.debug('Sending request to the Serene server to list octopi.')
        uri = self._uri

        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get octopus keys", e)

        self._handle_errors(r, "GET " + uri)

        return r.json()

    @staticmethod
    def _process_octopus_input(ssds,
                               name="",
                               description="",
                               feature_config=None,
                               model_type="randomForest",
                               resampling_strategy="NoResampling",
                               num_bags=10,
                               bag_size=10,
                               ontologies=None,
                               modeling_props=None):
        """Prepares a json string for octopus request object"""
        data = {
            "ssds": ssds,
            "name":name,
            "description": description,
            "modelType": model_type,
            "resamplingStrategy": resampling_strategy,
            "features": feature_config,
            "numBags": num_bags,
            "bagSize": bag_size,
            "ontologies": ontologies,
            "modelingProps": modeling_props
        }
        return {k: v for k, v in data.items() if v is not None}
