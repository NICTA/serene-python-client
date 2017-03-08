"""
License...
"""
import logging

from .api.session import Session
from .endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint
from .semantics import SSDInternal
from .elements import OctopusInternal

# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class Serene(object):
    """
        The Serene object is the main object. This holds a connection
        to the back-end and can create:

            1. Datasets - DataSet objects, built from uploaded csv files
            2. Ontologies - Ontology objects built from uploaded owl files with DataNodes and ClassNodes
            3. SSDs - Semantic Source Descriptions
            3. Octopii - An Octopus manages SSDs, Ontologies and can predict

        Ontologies can be passed to the SemanticModeller with
        add_ontology. This can be as an Ontology type or as an .owl
        file.

        To convert a csv file to a SemanticSourceDesc object,
        use to_ssd().

    """
    def __init__(self,
                 host='127.0.0.1',
                 port=8080,
                 auth=None,
                 cert=None,
                 trust_env=None):
        """
        Builds the SemanticModeller from a list of known
        ontologies and the config parameters.

        :param config: Arguments to configure the Schema Modeller
        :param ontologies: List of ontologies to use for mapping

        """
        self._session = Session(host, port, auth, cert, trust_env)

        self._datasets = DataSetEndpoint(self._session)

        self._ontologies = OntologyEndpoint(self._session)

        self._ssds = SSDEndpoint(self._session)

    def SSD(self, dataset, ontology):
        """
        Here we have the SSD class that the user can use to build SSDs.
        Note that we hide this in a nested class to pass the reference
        to the live server. Otherwise the user needs to pass in a session
        reference for a simple SSD() class. With the server reference
        we can use real dataset and ontology objects, rather than just
        IDs when talking to/from the server.
        """
        return SSDInternal(dataset, ontology, self._datasets, self._ontologies)

    class Octopus(OctopusInternal):
        def __init__(self):
            super().__init__(name, self._session)

        # # set the config args...
        # default_args = {
        #     "kmeans": 5,
        #     "threshold": 0.7,
        #     "search-depth": 100
        # }
        # self.config = default_args
        #
        # # fill in the default args...
        # if type(config) == dict:
        #     for k, v in config.items():
        #         if k in default_args.keys():
        #             self.config[k] = v
        #         else:
        #             _logger.warning("The key {} is not supported for config".format(k))

        # # add the ontology list...
        # self._ontologies = []
        # if ontologies is not None:
        #     for o in ontologies:
        #         self.add_ontology(Ontology(o))

        # _logger.debug("Config set to:", self.config)
        # _logger.debug("Ontologies set to: {}".format(self._ontologies))

    # def add_ontology(self, ontology):
    #     """
    #     Adds an ontology to the list. This method can accept Ontology
    #     type objects or .owl files.
    #
    #     :param ontology: The Ontology object to add, or a .owl filename
    #     :return: Updated SemanticModeller object
    #     """
    #     if type(ontology) == Ontology:
    #         self._ontologies.append(ontology)
    #     elif type(ontology) == str:
    #         # attempt to open from a file...
    #         self.add_ontology(Ontology(ontology))
    #     else:
    #         raise Exception("Failed to add ontology: {}".format(ontology))
    #
    #     return self

    # def to_ssd(self, filename):
    #     """
    #     Loads in a datafile to transform to an ssd object
    #
    #     :param filename: A CSV file to convert to an SSD
    #     :return: SemanticSourceDesc object
    #     """
    #     ssd = SSD(filename, self)
    #     return ssd

    @staticmethod
    def _flatten(xs):
        """Simple flatten from 2D to 1D iterable"""
        return [x for y in xs for x in y]

    # @property
    # def class_nodes(self):
    #     """All the class_nodes available in the ontologies"""
    #     return self.flatten(o.class_nodes for o in self._ontologies)
    #
    # @property
    # def data_nodes(self):
    #     """All the data_nodes available in the ontologies"""
    #     return self.flatten(o.data_nodes for o in self._ontologies)
    #
    # @property
    # def links(self):
    #     """All links currently present in the ontologies"""
    #     return self.flatten(o.links for o in self._ontologies)
    #
    # @property
    # def ontologies(self):
    #     """The read-only list of ontologies currently on the server"""
    #     return self._ontologies
    #def upload_ssd(self, ssd):
    #    pass

    @property
    def ontologies(self):
        return self._ontologies

    @property
    def datasets(self):
        return self._datasets

    @property
    def ssds(self):
        return self._ssds

    def __repr__(self):
        return "Serene({}:{}, {})".format(
            self._session.host,
            self._session.port,
            hex(id(self))
        )


