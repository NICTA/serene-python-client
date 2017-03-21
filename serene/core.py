"""
License...
"""
import logging
import os
import pandas as pd

from .api.session import Session
from .elements import Octopus, SSD
from .endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint, OctopusEndpoint, ModelEndpoint

# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.WARN)


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

        if self._session is None:
            msg = "Failed to initialize session at {}:{}".format(host, port)
            raise Exception(msg)

        self._datasets = DataSetEndpoint(self._session)

        self._ontologies = OntologyEndpoint(self._session)

        self._ssds = SSDEndpoint(self._session, self._datasets, self._ontologies)

        self._models = ModelEndpoint(self._session, self._datasets)

        self._octopii = OctopusEndpoint(self._session, self._models, self._ontologies, self._ssds)

    def SSD(self, dataset, ontology, name):
        """
        Here we have the SSD class that the user can use to build SSDs.
        Note that we hide this in a nested class to pass the reference
        to the live server. Otherwise the user needs to pass in a session
        reference for a simple SSD() class. With the server reference
        we can use real dataset and ontology objects, rather than just
        IDs when talking to/from the server.
        """
        return SSD(dataset, ontology, name)

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

    def load(self, ontology=None, datasets=None, map_file=None, link_file=None):
        """
        Load sets up the server with training data from files. The user must specify
        the list of datasets, the list of ontologies, a map file in the format:

        column,file,class
        column_name, filename, class.property
        column_name, filename, class.property
        column_name, filename, class.property

        and a link file in the following format

        file,src,link,dst
        filename, source_class, link_label, destination_class
        filename, source_class, link_label, destination_class
        filename, source_class, link_label, destination_class
        filename, source_class, link_label, destination_class

        :param ontology: A list of ontologies, or a single ontology
        :param datasets: A list of CSV datasets
        :param map_file: A CSV file with the column - data_node mappings as above
        :param link_file: A CSV file with the class-class links as above
        :return: (dataset list, ontology list, ssd list)
        """
        if type(ontology) == str:
            ontology = [ontology]

        uploaded_ontologies = []
        uploaded_datasets = []
        uploaded_ssds = []

        # first upload the ontologies...
        for o in ontology:
            if not os.path.exists(o):
                msg = "Ontology {} does not exist."
                raise ValueError(msg)
            uploaded_ontologies.append(self._ontologies.upload(o))

        # secondly upload the datasets....
        for d in datasets:
            if not os.path.exists(d):
                msg = "Dataset {} does not exist."
                raise ValueError(msg)
            uploaded_datasets.append(self._datasets.upload(d))

        # now build the ssds from the files...
        map_df = pd.read_csv(map_file)
        link_df = pd.read_csv(link_file)

        # check that the datasets match
        map_datasets = set(map_df['file'].unique())
        link_datasets = set(link_df['file'].unique())
        server_datasets = set([x.filename for x in uploaded_datasets])
        assert(map_datasets == server_datasets)
        assert(link_datasets == server_datasets)

        ssd_map = {ds.filename: SSD(ds, uploaded_ontologies)
                   for ds in uploaded_datasets}

        # first go through the map file...
        for _, row in map_df.iterrows():
            column = row['column']
            file = row['file']
            data_node = row['class']
            # now add the map...
            ssd_map[file].map(column, data_node)

        # first go through the map file...
        for _, row in link_df.iterrows():
            file = row['file']
            src = row['src']
            link = row['link']
            dst = row['dst']
            # now add the links...
            ssd_map[file].link(src, link, dst)

        # now attempt to upload all the ssds..
        for ssd in ssd_map.values():
            uploaded_ssds.append(self._ssds.upload(ssd))

        return uploaded_datasets, uploaded_ontologies, uploaded_ssds

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

    @property
    def session(self):
        return self._session

    def __repr__(self):
        return "Serene({}:{}, {})".format(
            self._session.host,
            self._session.port,
            hex(id(self))
        )
