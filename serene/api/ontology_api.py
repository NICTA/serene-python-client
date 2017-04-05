import logging
import random
import string
import tempfile
import os
from enum import unique, Enum
from urllib.parse import urljoin

import requests

from .exceptions import InternalError
from .http import HTTPObject


@unique
class OwlFormat(Enum):
    TURTLE = 'ttl'
    JSON = 'jsonld'
    XML = 'xml'
    RDFa = 'rdfa'
    Notation3 = 'n3'


class OntologyAPI(HTTPObject):
    """
    Handles the Ontology endpoint requests
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI of the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'owl/')
        self.OWL_FORMATS = {x.value for x in OwlFormat}

    def keys(self):
        """
        List ids of all ontologies in the repository at the Serene server.
        If the connection fails, empty list is returned and connection error is logged.

        Returns: list of ontology keys.
        Raises: InternalDIError on failure.
        """
        logging.debug('Sending request to the Serene server to list ontologies.')
        try:
            r = self.connection.get(self._uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("ontology keys", e)
        self._handle_errors(r, "GET " + self._uri)
        return r.json()

    def item(self, key):
        """
        Get information on a specific ontology from the repository on the Serene server.
        Args:
             key: integer which is the key of the Ontology.

        Returns: dictionary.
        """
        logging.debug('Sending request to Serene server to get the ontology info.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to list ontology", e)

        self._handle_errors(r, "GET " + uri)

        return r.json()

    @staticmethod
    def _gen_id():
        return ''.join(random.sample(string.ascii_uppercase, 10))

    def _create_local_owl_file(self, path):
        return open(path, "wb")

    def owl_file(self, key):
        """
        Get the actual owl file from the repository on the Serene server.
        Args:
             key: integer which is the key of the Ontology.

        Returns: dictionary.
        """
        logging.debug("Inferring extension pof the ontology...")
        owl_json = self.item(key)

        try:
            extension = os.path.splitext(owl_json["name"])[1]
        except:
            logging.debug("...setting default owl extension")
            extension = "owl"

        logging.debug('Sending request to Serene server to get the ontology file.')
        uri = "{}{}/file".format(self._uri, str(key))
        try:
            # extension of the ontology matters!
            path = os.path.join(
                tempfile.gettempdir(),
                "{}{}".format(self._gen_id(), extension))

            r = requests.get(uri, stream=True)

            if r.status_code == 200:
                with self._create_local_owl_file(path) as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            else:
                raise Exception("Failed to get ontology file. Status code: {}".format(r.url))
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to list ontology", e)

        return path

    @staticmethod
    def process_format(owl_format):
        """
        Helper function to convert from rdflib formats to server formats
        :param owl_format:
        :return:
        """
        if owl_format == "turtle":
            return "ttl"
        elif owl_format == "json-ld":
            return "jsonld"
        return owl_format

    def post(self, description, file_path, owl_format):
        """
        Post a new ontology to the Serene server.
        Args:
             description: string which describes the ontology to be posted
             file_path: string which indicates the location of the OWL file
             owl_format: type of ontology format
                     e.g. 'turtle', 'jsonld', 'rdfxml', 'owl'
        Returns: Dictionary.
        """

        logging.debug('Sending request to the schema matcher server to post a dataset.')
        uri = self._uri
        owl_format = self.process_format(owl_format)

        if owl_format not in self.OWL_FORMATS:
            msg = "Ontology format value {} is not supported. " \
                  "Use one of: {}".format(owl_format, self.OWL_FORMATS)
            raise ValueError(msg)
        try:
            f = {"file": open(file_path, "rb")}
            data = {
                "description": str(description),
                "format": owl_format
            }
            r = self.connection.post(uri, data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to create ontology", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def update(self, key, description=None, file_path=None, owl_format=None):
        """
        Update an existing Ontology in the repository on the Serene server.
        Args:
             key: integer ontology id
             description: string which describes the dataset to be posted
             type_map: dictionary with type map for the dataset

        :return:
        """
        logging.debug('Sending request to the schema matcher server to post a dataset.')
        uri = urljoin(self._uri, str(key))

        if owl_format is not None and owl_format not in self.OWL_FORMATS:
            msg = "Ontology format value {} is not supported. " \
                  "Use one of: {}".format(owl_format, self.OWL_FORMATS)
            raise ValueError(msg)
        try:
            if file_path is None:
                f = None
            else:
                f = {"file": open(file_path, "rb")}

            data = {
                "description": str(description) if description is not None else None,
                "format": owl_format
            }
            data = {k: v for k, v in data.items() if v is not None}

            r = self.connection.post(uri, data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to update ontology", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def delete(self, key):
        """
        Delete a specific ontology from the repository at the Serene server.
        Args:
             key: int

        Returns:
        """
        logging.debug('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self._uri, str(key))

        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("delete_ontology", e)

        self._handle_errors(r, "DELETE " + uri)

        return r.json()
