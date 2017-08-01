"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Test class for the IntegrationGraph which uses the chuffed solver
"""

# import unittest2 as unittest
import os
import logging
import tarfile
import csv
import time

from .utils import TestWithServer
from serene import Status
from serene.elements import SSD
from serene.endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint, OctopusEndpoint, ModelEndpoint
from serene.elements.semantics.base import KARMA_DEFAULT_NS
from stp.integration import IntegrationGraph


# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class TestIntegrationGraph(TestWithServer):
    """

    """
    TRAIN_NUM = 3

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        self._benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                            "tests", "resources", "museum_benchmark")
        self._museum_owl_dir = os.path.join(self._benchmark_path, "owl")
        self._museum_data = os.path.join(self._benchmark_path, 'dataset')
        self._museum_ssd = os.path.join(self._benchmark_path, 'ssd')

    def setUp(self):
        self._clear_storage()

        _logger.debug("SetUp complete")

    def tearDown(self):
        """
        Be sure to remove all storage elements once a test is finished...
        :return:
        """
        self._clear_storage()

    def _add_owls(self):
        # add ontologies
        ontologies = []
        for path in os.listdir(self._museum_owl_dir):
            f = os.path.join(self._museum_owl_dir, path)
            ontologies.append(self._serene._ontologies.upload(f))
        return ontologies

    def _add_datasets(self, ontologies):
        # we need to unzip first the data
        if os.path.exists(os.path.join(self._museum_data, "data.tar.gz")):
            with tarfile.open(os.path.join(self._museum_data, "data.tar.gz")) as f:
                f.extractall(path=self._museum_data)

        # add datasets with their ssds
        for ds in os.listdir(self._museum_data):
            if ds.endswith(".gz"):
                continue # skip archive
            ds_f = os.path.join(self._museum_data, ds)
            ds_name = os.path.splitext(ds)[0]
            ssd_f = os.path.join(self._museum_ssd, ds_name + ".ssd")

            _logger.debug("Adding dataset: {}".format(ds_f))
            dataset = self._serene._datasets.upload(ds_f, description="museum_benchmark")

            _logger.debug("Adding ssd: {}".format(ssd_f))
            new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
            empty_ssd = SSD(dataset, ontologies)
            ssd = empty_ssd.update(new_json, self._serene._datasets, self._serene._ontologies)
            self._serene._ssds.upload(ssd)
            # we remove the csv dataset
            os.remove(ds_f)

    def create_octopus(self, ontologies):
        ssds = self._serene._ssds.items[:self.TRAIN_NUM]
        octo_local = self._serene.Octopus(
            ssds=ssds,
            ontologies=ontologies,
            name='octopus-unknown',
            description='Testing example for places and companies',
            resampling_strategy="NoResampling",  # optional
            num_bags=80,  # optional
            bag_size=50,  # optional
            model_type="randomForest",
            modeling_props={
                "compatibleProperties": True,
                "ontologyAlignment": True,
                "addOntologyPaths": False,
                "mappingBranchingFactor": 50,
                "numCandidateMappings": 10,
                "topkSteinerTrees": 50,
                "multipleSameProperty": False,
                "confidenceWeight": 1.0,
                "coherenceWeight": 1.0,
                "sizeWeight": 0.5,
                "numSemanticTypes": 10,
                "thingNode": False,
                "nodeClosure": True,
                "propertiesDirect": True,
                "propertiesIndirect": True,
                "propertiesSubclass": True,
                "propertiesWithOnlyDomain": True,
                "propertiesWithOnlyRange": True,
                "propertiesWithoutDomainRange": False,
                "unknownThreshold": 0.05
            },
            feature_config={
                "activeFeatures": [
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
                    "entropy-for-discrete-values",
                    "shannon-entropy"
                ]
                ,
                "activeFeatureGroups": [
                    "char-dist-features",
                    "inferred-data-type",
                    "stats-of-text-length",
                    "stats-of-numeric-type"
                    , "prop-instances-per-class-in-knearestneighbours"
                    # ,"mean-character-cosine-similarity-from-class-examples"
                    # ,"min-editdistance-from-class-examples"
                    # ,"min-wordnet-jcn-distance-from-class-examples"
                    # ,"min-wordnet-lin-distance-from-class-examples"
                ],
                "featureExtractorParams": [
                    {
                        "name": "prop-instances-per-class-in-knearestneighbours",
                        "num-neighbours": 3
                    }
                    , {
                        "name": "min-editdistance-from-class-examples",
                        "max-comparisons-per-class": 3
                    }, {
                        "name": "min-wordnet-jcn-distance-from-class-examples",
                        "max-comparisons-per-class": 3
                    }, {
                        "name": "min-wordnet-lin-distance-from-class-examples",
                        "max-comparisons-per-class": 3
                    }
                ]
            }
        )

        # add this to the endpoint...
        print("Now we upload to the server")
        octo = self._serene.octopii.upload(octo_local)

        start = time.time()
        print()
        print("Next we can train the Octopus")
        print("The initial state for {} is {}".format(octo.id, octo.state))
        print("Training...")
        octo.train()
        print("Done in: {}".format(time.time() - start))
        print("The final state for {} is {}".format(octo.id, octo.state))

        return octo

    def _clear_storage(self):
        """Removes all server elements"""
        for oc in self._serene.octopii.items:
            self._serene.octopii.remove(oc)

        for oc in self._serene.models.items:
            self._serene.models.remove(oc)

        for ssd in self._serene.ssds.items:
            self._serene.ssds.remove(ssd)

        for ds in self._serene.datasets.items:
            self._serene.datasets.remove(ds)

        for on in self._serene.ontologies.items:
            self._serene.ontologies.remove(on)

    def test_get_initial_alignment(self):
        ontologies = self._add_owls()
        self._add_datasets(ontologies)
        self.assertEqual(len(self._serene._ssds.items), 29)
        self.assertEqual(len(self._serene._datasets.items), 29)
        self.assertEqual(len(self._serene._ontologies.items), 11)

        octopus = self.create_octopus(ontologies)
        self.assertEqual(octopus.state.status, Status.COMPLETE)

        ds = self._serene.datasets.items[0]
        integrat = IntegrationGraph(octopus=octopus, dataset=ds,
                                    match_threshold=0.01,
                                    simplify_graph=True,
                                    patterns=None)

        print("# nodes", integrat.graph.number_of_nodes(), ", #edges:", integrat.graph.number_of_edges())
        self.assertEqual(integrat.graph.number_of_nodes(), 10)
        self.assertEqual(integrat.graph.number_of_edges(), 10)

    def test_read_config(self):
        self.fail()

    def test_add_matches(self):
        self.fail()

    def test__alignment2dzn(self):
        self.fail()

    def test__matches2dzn(self):
        self.fail()

    def test__generate_oznfzn(self):
        self.fail()

    def test__find_link(self):
        self.fail()

    def test__digraph_to_ssd(self):
        self.fail()

    def test__process_solution(self):
        self.fail()

    def test_solve(self):
        self.fail()
