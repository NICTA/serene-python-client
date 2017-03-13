import logging
from urllib.parse import urljoin

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
             name=None,
             description=None,
             features=None,
             resampling_strategy=None,
             num_bags=None,
             bag_size=None,
             ontologies=None,
             ssds=None,
             modeling_properties=None):
        """
        Post a new Octopus to the Serene server.

        :param name: The name of the Octopus
        :param description: A descriptive text
        :param features: The ML features to use for the schema matcher, options
                         { "activeFeatures" : [
                              "num-unique-vals",
                              "prop-unique-vals",
                              "prop-missing-vals",
                              "ratio-alpha-chars",
                              "prop-numerical-chars",
                              "prop-whitespace-chars",
                              "prop-entries-with-at-sign",
                              "prop-entries-with-hyphen",
                              "prop-range-format",
                              "is-discrete",
                              "entropy-for-discrete-values"
                            ],
                            "activeFeatureGroups" : [
                              "inferred-data-type",
                              "stats-of-text-length",
                              "stats-of-numeric-type",
                              "prop-instances-per-class-in-knearestneighbours",
                              "mean-character-cosine-similarity-from-class-examples",
                              "min-editdistance-from-class-examples",
                              "min-wordnet-jcn-distance-from-class-examples",
                              "min-wordnet-lin-distance-from-class-examples"
                            ],
                            "featureExtractorParams" : [
                                 {
                                  "name" : "prop-instances-per-class-in-knearestneighbours",
                                  "num-neighbours" : 3
                                 }, {
                                  "name" : "min-editdistance-from-class-examples",
                                  "max-comparisons-per-class" : 20
                                 }, {
                                  "name" : "min-wordnet-jcn-distance-from-class-examples",
                                  "max-comparisons-per-class" : 20
                                 }, {
                                  "name" : "min-wordnet-lin-distance-from-class-examples",
                                  "max-comparisons-per-class" : 20
                                 }
                            ]
                         }
        :param resampling_strategy: The sampling strategy to use to balance the classes:
                                    "UpsampleToMax"
                                    "ResampleToMean"
                                    "Bagging"
                                    "UpsampleToMean"
                                    "BaggingToMax"
                                    "BaggingToMean"
                                    "NoResampling"
        :param num_bags: The number of bags for the Bagging resampling strategies
        :param bag_size: The size of the bags for the Bagging resampling strategies
        :param ontologies: The list of Ontology keys used in this Octopus
        :param ssds: The list of SSD keys used in this Octopus
        :param modeling_properties:

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
                                             resampling_strategy)
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
               resampling_strategy=None):
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
                                             resampling_strategy)

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
                             resampling_strategy=None):
        """Prepares a json update string for the model update request"""
        data = {
            "description": description,
            "classes": classes,
            "modelType": model_type,
            "labelData": {str(key): str(val) for key, val in labels.items()},
            "costMatrix": cost_matrix,
            "resamplingStrategy": resampling_strategy,
            "features": feature_config
        }
        return {k: v for k, v in data.items() if v is not None}