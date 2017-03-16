"""
License...
"""
import logging

from .api.session import Session
from .elements import Octopus, SSD
from .endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint, OctopusEndpoint, ModelEndpoint

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
            4. Octopii - An Octopus manages SSDs, Ontologies and can predict
            5. Models - The predictive model used by the Octopii.

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

        self._octopii = OctopusEndpoint(self._session)

        self._models = ModelEndpoint(self._session)

    @staticmethod
    def SSD(dataset, ontology):
        """
        Here we have the SSD class that the user can use to build SSDs.
        Note that we hide this in a nested class to pass the reference
        to the live server. Otherwise the user needs to pass in a session
        reference for a simple SSD() class. With the server reference
        we can use real dataset and ontology objects, rather than just
        IDs when talking to/from the server.
        """
        return SSD(dataset, ontology)

    @staticmethod
    def Octopus(ssds,
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
        Here we have the Octopus class that the user can use to build an
        Octopus object.
        Note that we hide this in a def to pass the reference
        to the live server. Otherwise the user needs to pass in a session
        reference for a simple Octopus() class. With the server reference
        we can use real ssd and ontology objects, rather than just
        IDs when talking to/from the server.
        """
        return Octopus(
            ssds=ssds,
            name=name,
            description=description,
            feature_config=feature_config,
            model_type=model_type,
            resampling_strategy=resampling_strategy,
            num_bags=num_bags,
            bag_size=bag_size,
            ontologies=ontologies,
            modeling_props=modeling_props
        )

    @property
    def ontologies(self):
        return self._ontologies

    @property
    def datasets(self):
        return self._datasets

    @property
    def ssds(self):
        return self._ssds

    @property
    def octopii(self):
        return self._octopii

    @property
    def models(self):
        return self._models

    def __repr__(self):
        return "Serene({}:{}, {})".format(
            self._session.host,
            self._session.port,
            hex(id(self))
        )
