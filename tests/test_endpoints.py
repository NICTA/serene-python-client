"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import datetime
import os
import sys
from io import StringIO

from serene.elements.elements import ClassNode, Column, DataNode
from serene.elements.octopus import Octopus
from serene.elements.semantics.ontology import Ontology
from serene.elements.semantics.ssd import SSD
from serene.endpoints import (DataSetEndpoint, ModelEndpoint, OctopusEndpoint,
                              OntologyEndpoint, SSDEndpoint)
from tests.utils import TestWithServer
from serene.api.http import BadRequestError


class TestDataSetEndpoint(TestWithServer):
    """
    Tests the dataset endpoint
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        path = os.path.join(os.path.dirname(__file__), "resources")
        self._test_file = os.path.join(path, 'data', 'businessInfo.csv')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._clear_datasets()
        assert(os.path.isfile(self._test_file))

    def _clear_datasets(self):
        """Removes all datasets"""
        for ds in self._datasets.items:
            self._datasets.remove(ds)

    def tearDown(self):
        """
        Be sure to remove all datasets once a test is finished...
        :return:
        """
        self._clear_datasets()

    def test_upload(self):
        """
        Tests the uploading of a dataset
        :return:
        """
        self.assertEqual(len(self._datasets.items), 0)

        # only a dataset string path should be allowed
        with self.assertRaises(Exception):
            self._datasets.upload(1234)

        # only an existing dataset string path should be allowed
        with self.assertRaises(Exception):
            self._datasets.upload('resources/data/doesnt-exist.csv')

        # now upload a nice dataset
        d = self._datasets.upload(self._test_file)

        self.assertEqual(len(self._datasets.items), 1)
        self.assertEqual(d.filename, os.path.basename(self._test_file))
        self.assertEqual(
            [c.name for c in d.columns],
            ['company', 'ceo', 'city', 'state']
        )

    def test_remove(self):
        """
        Tests the removal of a dataset
        :return:
        """
        self.assertEqual(len(self._datasets.items), 0)

        # upload some datasets
        d1 = self._datasets.upload(self._test_file)
        d2 = self._datasets.upload(self._test_file)
        self.assertEqual(len(self._datasets.items), 2)

        # now let's remove one
        self._datasets.remove(d1)
        self.assertEqual(len(self._datasets.items), 1)
        self.assertEqual(self._datasets.items[0].id, d2.id)

        # let's remove the other by id
        self._datasets.remove(d2.id)
        self.assertEqual(len(self._datasets.items), 0)

    def test_items(self):
        """
        Tests item manipulation for the dataset list
        :return:
        """
        self.assertEqual(len(self._datasets.items), 0)

        # upload some datasets
        d1 = self._datasets.upload(self._test_file)
        d2 = self._datasets.upload(self._test_file)
        self.assertEqual(len(self._datasets.items), 2)

        # now let's remove one
        self._datasets.remove(d1)
        self.assertEqual(len(self._datasets.items), 1)

        # make sure the right one remains
        self.assertEqual(self._datasets.items[0], d2)

        # now let's remove the last one
        self._datasets.remove(d2)
        self.assertEqual(len(self._datasets.items), 0)


class TestOntologyEndpoint(TestWithServer):
    """
    Tests the ontology endpoint
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        path = os.path.join(os.path.dirname(__file__), "resources")
        self._test_file = os.path.join(path, 'owl', 'dataintegration_report_ontology.ttl')
        self._bad_params_file = os.path.join(path, 'owl', 'bad-params.ttl')
        self._ontologies = None

    def setUp(self):
        self._ontologies = OntologyEndpoint(self._session)
        self._clear_ontologies()
        assert(os.path.isfile(self._test_file))

    def _clear_ontologies(self):
        """Removes all ontologies"""
        for on in self._ontologies.items:
            self._ontologies.remove(on)

    def tearDown(self):
        """
        Be sure to remove all datasets once a test is finished...
        :return:
        """
        self._clear_ontologies()

    def test_upload(self):
        """
        Tests the uploading of a ontology
        :return:
        """
        description_string = "description-string"

        self.assertEqual(len(self._ontologies.items), 0)

        # now upload a nice ontology
        on = self._ontologies.upload(
            self._test_file,
            description=description_string,
            owl_format='ttl'
        )

        self.assertEqual(len(self._ontologies.items), 1)
        self.assertEqual(on.name, os.path.basename(self._test_file))
        self.assertEqual(on.description, description_string)
        self.assertEqual(len(on.class_nodes), 6)

    def test_bad_param_upload(self):
        """
        Tests that bad parameter ontologies raise exceptions
        :return:
        """
        description_string = "description-string"

        # only a ontology string path should be allowed
        with self.assertRaises(Exception):
            self._ontologies.upload(1234)

        # only an existing ontology string path should be allowed
        with self.assertRaises(Exception):
            self._ontologies.upload('resources/owl/doesnt-exist.owl')

        with self.assertRaises(Exception):
            # now upload a bad format ontology
            self._ontologies.upload(
                self._test_file,
                description=description_string,
                owl_format='junk'
            )

        with self.assertRaises(Exception):
            # now upload a bad param owl file
            self._ontologies.upload(
                self._bad_params_file,
                description=description_string,
                owl_format='junk'
            )

    def test_remove(self):
        """
        Tests the removal of a ontology
        :return:
        """
        self.assertEqual(len(self._ontologies.items), 0)

        # upload some ontologies
        o1 = self._ontologies.upload(self._test_file)
        o2 = self._ontologies.upload(self._test_file)
        self.assertEqual(len(self._ontologies.items), 2)

        # now let's remove one
        self._ontologies.remove(o1)
        self.assertEqual(len(self._ontologies.items), 1)
        self.assertEqual(self._ontologies.items[0].id, o2.id)

        # let's remove the other by id
        self._ontologies.remove(o2.id)
        self.assertEqual(len(self._ontologies.items), 0)

    def test_items(self):
        """
        Tests item manipulation for the ontology list
        :return:
        """
        self.assertEqual(len(self._ontologies.items), 0)

        # upload some ontology
        o1 = self._ontologies.upload(self._test_file)
        o2 = self._ontologies.upload(self._test_file)
        self.assertEqual(len(self._ontologies.items), 2)

        # now let's remove one
        self._ontologies.remove(o1)
        self.assertEqual(len(self._ontologies.items), 1)

        # make sure the right one remains
        self.assertEqual(self._ontologies.items[0].id, o2.id)

        # now let's remove the last one
        self._ontologies.remove(o2)
        self.assertEqual(len(self._ontologies.items), 0)


def initSsdEndpoint(session):
    result = {}
    result["datasetEndpoint"] = DataSetEndpoint(session)
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "resources",
        "data",
        "businessInfo.csv")
    result["dataset"] = result["datasetEndpoint"].upload(dataset_path)

    result["ontologyEndpoint"] = OntologyEndpoint(session)
    result["ontology"] = result["ontologyEndpoint"].upload(Ontology()
        .uri("http://www.semanticweb.org/serene/example_ontology")
        .owl_class("Place", ["name", "postalCode"])
        .owl_class("City", is_a="Place")
        .owl_class("Person", {"name": str, "phone": int, "birthDate": datetime.datetime})
        .owl_class("Organization", {"name": str, "phone": int, "email": str})
        .owl_class("Event", {"startDate": datetime.datetime, "endDate": datetime.datetime})
        .owl_class("State", is_a="Place")
        .link("Person", "bornIn", "Place")
        .link("Organization", "ceo", "Person")
        .link("Place", "isPartOf", "Place")
        .link("Person", "livesIn", "Place")
        .link("Event", "location", "Place")
        .link("Organization", "location", "Place")
        .link("Organization", "operatesIn", "City")
        .link("Place", "nearby", "Place")
        .link("Event", "organizer", "Person")
        .link("City", "state", "State")
        .link("Person", "worksFor", "Organization"))

    result["ssdEndpoint"] = SSDEndpoint(
        session,
        result["datasetEndpoint"],
        result["ontologyEndpoint"])

    result["localSsd"] = (
        SSD(result["dataset"], result["ontology"], name="business-info")
            .map(Column("company"), DataNode(ClassNode("Organization"), "name"))
            .map(Column("ceo"), DataNode(ClassNode("Person"), "name"))
            .map(Column("city"), DataNode(ClassNode("City"), "name"))
            .map(Column("state"), DataNode(ClassNode("State"), "name"))
            .link("Organization", "operatesIn", "City")
            .link("Organization", "ceo", "Person")
            .link("City", "state", "State")
    )

    return result


class TestSsdEndpoint(TestWithServer):
    def setUp(self):
        init_result = initSsdEndpoint(self._session)
        self.datasetEndpoint = init_result["datasetEndpoint"]
        self.dataset = init_result["dataset"]
        self.ontologyEndpoint = init_result["ontologyEndpoint"]
        self.ontology = init_result["ontology"]
        self.ssdEndpoint = init_result["ssdEndpoint"]
        self.ssd = init_result["localSsd"]

    def tearDown(self):
        for item in self.ssdEndpoint.items:
            self.ssdEndpoint.remove(item)

        for item in self.ontologyEndpoint.items:
            self.ontologyEndpoint.remove(item)

        for item in self.datasetEndpoint.items:
            self.datasetEndpoint.remove(item)

    def test_upload(self):
        ssd = self.ssdEndpoint.upload(self.ssd)

        self.assertEqual(len(ssd.class_nodes), 4)
        self.assertEqual(len(ssd.data_nodes), 4)
        self.assertEqual(len(ssd.data_links), 4)
        self.assertEqual(len(ssd.columns), 4)
        self.assertEqual(len(ssd.object_links), 3)
        self.assertEqual(len(ssd.mappings), 4)
        self.assertEqual(ssd.dataset, self.dataset)
        self.assertEqual(len(ssd.ontology), 1)

    def test_compare(self):
        ssd = self.ssdEndpoint.upload(self.ssd)
        result = self.ssdEndpoint.compare(ssd, ssd)
        self.assertEqual(
            result,
            {"precision": 1.0, "recall": 1.0, "jaccard": 1.0})

    def test_remove(self):
        ssd = self.ssdEndpoint.upload(self.ssd)
        self.ssdEndpoint.remove(ssd)
        self.assertEqual(len(self.ssdEndpoint.items), 0)

    def test_show(self):
        output = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = output
        try:
            ssd = self.ssdEndpoint.upload(self.ssd)
            self.ssdEndpoint.show()
        finally:
            sys.stdout = sys_stdout

        self.assertGreater(len(output.getvalue()), 0)

    def test_get(self):
        ssd = self.ssdEndpoint.upload(self.ssd)
        ssd_got = self.ssdEndpoint.get(ssd.id)

        self.assertEqual(ssd.id, ssd_got.id)

    def test_items(self):
        self.ssdEndpoint.upload(self.ssd)
        self.ssdEndpoint.upload(self.ssd)

        self.assertEqual(len(self.ssdEndpoint.items), 2)


def initOctopusEndpoint(session):
    result = initSsdEndpoint(session)

    result["ssd"] = result["ssdEndpoint"].upload(result["localSsd"])

    result["modelEndpoint"] = ModelEndpoint(session, result["datasetEndpoint"])
    result["octopusEndpoint"] = OctopusEndpoint(
        session,
        result["datasetEndpoint"],
        result["modelEndpoint"],
        result["ontologyEndpoint"],
        result["ssdEndpoint"])

    feature_config = {
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
            "entropy-for-discrete-values"
        ],
        "activeFeatureGroups": [
            "inferred-data-type",
            "stats-of-text-length",
            "stats-of-numeric-type",
            "prop-instances-per-class-in-knearestneighbours",
            "mean-character-cosine-similarity-from-class-examples",
            "min-editdistance-from-class-examples",
            "min-wordnet-jcn-distance-from-class-examples",
            "min-wordnet-lin-distance-from-class-examples"
        ],
        "featureExtractorParams": [
            {
                "name": "prop-instances-per-class-in-knearestneighbours",
                "num-neighbours": 3
            }, {
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

    result["localOctopus"] = Octopus(
        ssds=[result["ssd"]],
        name="octopus test",
        description="no description for a test",
        model_type="randomForest",
        resampling_strategy="NoResampling",
        num_bags=100,
        bag_size=10,
        ontologies=[result["ontology"]],
        modeling_props={},
        feature_config=feature_config
    )

    result["localInvalidOctopus"] = Octopus(
        ssds=[result["ssd"]],
        name="octopus test",
        description="no description for a test",
        model_type="randomForest",
        resampling_strategy="NoResampling",
        num_bags=100,
        bag_size=10,
        ontologies=[result["ontology"]],
        modeling_props={"topkSteinerTrees": -1},
        feature_config=feature_config
    )

    return result


class TestOctopusEndpoint(TestWithServer):
    def setUp(self):
        init_result = initOctopusEndpoint(self._session)
        self.ssdEndpoint = init_result["ssdEndpoint"]
        self.ontologyEndpoint = init_result["ontologyEndpoint"]
        self.datasetEndpoint = init_result["datasetEndpoint"]
        self.modelEndpoint = init_result["modelEndpoint"]
        self.octopusEndpoint = init_result["octopusEndpoint"]
        self.octopus = init_result["localOctopus"]
        self.invalidOctopus = init_result["localInvalidOctopus"]

    def tearDown(self):
        for item in self.octopusEndpoint.items:
            self.octopusEndpoint.remove(item)

        for item in self.ssdEndpoint.items:
            self.ssdEndpoint.remove(item)

        for item in self.ontologyEndpoint.items:
            self.ontologyEndpoint.remove(item)

        for item in self.datasetEndpoint.items:
            self.datasetEndpoint.remove(item)

    def test_upload(self):
        octopus = self.octopusEndpoint.upload(self.octopus)
        self.assertIsNotNone(octopus.id)
        self.assertEqual(octopus.name, self.octopus.name)
        self.assertEqual(octopus.description, self.octopus.description)
        self.assertEqual(octopus.feature_config, self.octopus.feature_config)
        self.assertEqual(octopus.model_type, self.octopus.model_type)
        self.assertEqual(octopus.resampling_strategy, self.octopus.resampling_strategy)
        self.assertEqual(octopus.num_bags, self.octopus.num_bags)
        self.assertEqual(octopus.bag_size, self.octopus.bag_size)
        self.assertEqual(octopus.ontologies[0].id, self.octopus.ontologies[0].id)
        self.assertGreater(len(octopus.modeling_props.keys()), 0)

    def test_upload_with_invalid_modeling_props(self):
        def upload():
            self.octopusEndpoint.upload(self.invalidOctopus)

        self.assertRaises(ValueError, upload)

    def test_remove(self):
        octopus = self.octopusEndpoint.upload(self.octopus)
        self.octopusEndpoint.remove(octopus)

        self.assertEqual(len(self.octopusEndpoint.items), 0)

    def test_show(self):
        output = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = output
        try:
            self.octopusEndpoint.upload(self.octopus)
            self.octopusEndpoint.show()
        finally:
            sys.stdout = sys_stdout

        self.assertGreater(len(output.getvalue()), 0)

    def test_get(self):
        octopus = self.octopusEndpoint.upload(self.octopus)
        octopus_got = self.octopusEndpoint.get(octopus.id)

        self.assertEqual(octopus.id, octopus_got.id)

    def test_items(self):
        self.octopusEndpoint.upload(self.octopus)
        self.octopusEndpoint.upload(self.octopus)

        self.assertEqual(len(self.octopusEndpoint.items), 2)


class TestModelEndpoint(TestWithServer):
    def setUp(self):
        init_result = initOctopusEndpoint(self._session)
        self.ssdEndpoint = init_result["ssdEndpoint"]
        self.ontologyEndpoint = init_result["ontologyEndpoint"]
        self.datasetEndpoint = init_result["datasetEndpoint"]
        self.modelEndpoint = init_result["modelEndpoint"]
        self.octopusEndpoint = init_result["octopusEndpoint"]
        self.octopus = self.octopusEndpoint.upload(init_result["localOctopus"])

    def tearDown(self):
        for item in self.octopusEndpoint.items:
            self.octopusEndpoint.remove(item)

        for item in self.ssdEndpoint.items:
            self.ssdEndpoint.remove(item)

        for item in self.ontologyEndpoint.items:
            self.ontologyEndpoint.remove(item)

        for item in self.datasetEndpoint.items:
            self.datasetEndpoint.remove(item)

    def test_remove(self):
        def remove():
            self.modelEndpoint.remove(self.octopus.matcher_id)

        self.assertRaises(BadRequestError, remove)

    def test_show(self):
        output = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = output
        try:
            self.modelEndpoint.show()
        finally:
            sys.stdout = sys_stdout

        self.assertGreater(len(output.getvalue()), 0)

    def test_get(self):
        model = self.modelEndpoint.get(self.octopus.matcher_id)

        self.assertEqual(model.id, self.octopus.matcher_id)

    def test_items(self):
        self.assertEqual(len(self.modelEndpoint.items), 1)
