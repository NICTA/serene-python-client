"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the octopus module
"""
import json
import datetime
import os
import unittest2 as unittest

from serene.elements import Column
from serene import Octopus, Ontology, DataNode, ClassNode, ModelState, SamplingStrategy, ModelType, Status
from ..utils import TestWithServer


class TestOctopus(TestWithServer):
    """
    Tests the Octopus class
    """
    def __init__(self, methodName='runTest'):
        """
        Initialization for the Octopus tester
        """
        super().__init__(methodName)

        # print("current dir: ", os.path.abspath(os.curdir))
        pp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "octopus", "octopus-test.json")
        with open(pp) as f:
            self.octo_json = json.load(f)

        self.datasets = None
        self.ontology = None
        self.ssds = None
        self.octo_local = None

    def _datasets(self):
        """
        Load up the example datasets
        :return:
        """
        ds = self._serene.datasets

        # data_path = os.path.join("tests", "resources", "data")
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "data")
        business_info = ds.upload(os.path.join(data_path, 'businessInfo.csv'))
        employee_address = ds.upload(os.path.join(data_path, 'EmployeeAddressesSingleName.csv'))
        get_cities = ds.upload(os.path.join(data_path, 'getCities.csv'))
        get_employees = ds.upload(os.path.join(data_path, 'getEmployees.csv'))
        postal_code = ds.upload(os.path.join(data_path, 'postalCodeLookup.csv'))

        return [business_info, employee_address, get_cities, get_employees, postal_code]

    def _ontology(self):
        """
        Load up test ontology...
        :return:
        """
        on = self._serene.ontologies
        owl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "owl")
        return on.upload(Ontology(os.path.join(owl_path, 'dataintegration_report_ontology.ttl')))

    def _ssds(self, datasets, ontology):
        """
        Load up the example SSDs
        :param datasets:
        :param ontology:
        :return:
        """
        business_info_ssd = (self._serene
                             .SSD(datasets[0], ontology, name='business-info')
                             .map(Column("company"), DataNode(ClassNode("Organization"), "name"))
                             .map(Column("ceo"), DataNode(ClassNode("Person"), "name"))
                             .map(Column("city"), DataNode(ClassNode("City"), "name"))
                             .map(Column("state"), DataNode(ClassNode("State"), "name"))
                             .link("Organization", "operatesIn", "City")
                             .link("Organization", "ceo", "Person")
                             .link("City", "state", "State"))

        employee_address_ssd = (self._serene
                                .SSD(datasets[1], ontology, name='employee-addr')
                                .map("name", "Person.name")
                                .map("address", "Place.name")
                                .map("postcode", "Place.postalCode")
                                .link("Person", "livesIn", "Place"))

        get_cities_ssd = (self._serene
                          .SSD(datasets[2], ontology, name='cities')
                          .map(Column("city"), DataNode(ClassNode("City"), "name"))
                          .map(Column("state"), DataNode(ClassNode("State"), "name"))
                          .link("City", "state", "State"))

        get_employees_ssd = (self._serene
                             .SSD(datasets[3], ontology, name='employees')
                             .map(Column("employer"), DataNode(ClassNode("Organization"), "name"))
                             .map(Column("employee"), DataNode(ClassNode("Person"), "name"))
                             .link("Person", "worksFor", "Organization"))

        postal_code_ssd = (self._serene
                           .SSD(datasets[4], ontology, name='postal-code')
                           .map(Column("zipcode"), "Place.postalCode")
                           .map(Column("city"), "City.name")
                           .map(Column("state"), "State.name")
                           .link("City", "state", "State")
                           .link("City", "isPartOf", "Place"))
        ssds = [business_info_ssd, employee_address_ssd, get_cities_ssd,
                get_employees_ssd, postal_code_ssd]

        for ssd in ssds:
            self._serene.ssds.upload(ssd)

        return ssds

    def setUp(self):
        print("current dir: ", os.path.abspath(os.curdir))
        self.datasets = self._datasets()
        self.ontology = self._ontology()
        self.ssds = self._ssds(self.datasets, self.ontology)

        # upload all these to the server. Here we ignore the return value (the SSD object will be updated)
        self.octo_local = Octopus(
            ssds=self.ssds,
            ontologies=[self.ontology],
            name='octopus-test',
            description='Testing example for places and companies',
            resampling_strategy="NoResampling",  # optional
            num_bags=100,  # optional
            bag_size=10,  # optional
            model_type="randomForest",
            modeling_props={
                "compatibleProperties": True,
                "ontologyAlignment": False,
                "addOntologyPaths": False,
                "mappingBranchingFactor": 50,
                "numCandidateMappings": 10,
                "topkSteinerTrees": 10,
                "multipleSameProperty": False,
                "confidenceWeight": 1,
                "coherenceWeight": 1,
                "sizeWeight": 0.5,
                "numSemanticTypes": 4,
                "thingNode": False,
                "nodeClosure": True,
                "propertiesDirect": True,
                "propertiesIndirect": True,
                "propertiesSubclass": True,
                "propertiesWithOnlyDomain": True,
                "propertiesWithOnlyRange": True,
                "propertiesWithoutDomainRange": False
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
        )

    def tearDown(self):
        for o in self._serene.octopii.items:
            self._serene.octopii.remove(o)

        for ssd in self._serene.ssds.items:
            self._serene.ssds.remove(ssd)

        for ds in self._serene.datasets.items:
            self._serene.datasets.remove(ds)

        for o in self._serene.ontologies.items:
            self._serene.ontologies.remove(o)

    def _octopus(self):
        """
        Upload an octopus object...
        :return:
        """
        return self._serene.octopii.upload(self.octo_local)

    def test_resampling(self):
        """
        Tests that the resampling args are protected
        :return:
        """
        types = {"UpsampleToMax",
                 "ResampleToMean",
                 "UpsampleToMean",
                 "Bagging",
                 "BaggingToMax",
                 "BaggingToMean",
                 "NoResampling"}

        self.assertSetEqual(types, set(SamplingStrategy.values()))

        for t in types:
            o = Octopus(resampling_strategy=t)
            self.assertEqual(o.resampling_strategy, t)

        with self.assertRaises(ValueError):
            Octopus(resampling_strategy="junk")

    def test_model_type(self):
        """
        Tests that the modeling args are protected
        :return:
        """
        types = {"randomForest"}
        self.assertSetEqual(types, set(ModelType.values()))

        for t in types:
            o = Octopus(model_type=t)
            self.assertEqual(o.model_type, t)

        with self.assertRaises(ValueError):
            Octopus(model_type="junk")

    def test_update(self):
        """
        Tests that the update is passing values correctly...
        :return:
        """
        server_octo = self._octopus()

        # first just update the json with the specifics...
        self.octo_json['id'] = server_octo.id
        self.octo_json['ssds'] = [s.id for s in self.ssds]
        self.octo_json['ontologies'] = [self.ontology.id]
        self.octo_json['lobsterID'] = server_octo.matcher_id

        o = server_octo.update(self.octo_json,
                               self._serene.session,
                               self._serene.datasets,
                               self._serene.models,
                               self._serene.ontologies,
                               self._serene.ssds)
        self.assertEqual(o.id, server_octo.id)
        self.assertEqual(o.stored, True)
        self.assertEqual(o.date_created, datetime.datetime(2017, 3, 31, 14, 9, 9, 668000))
        self.assertEqual(o.date_modified, datetime.datetime(2017, 3, 31, 14, 9, 52, 3000))
        self.assertEqual([ssd.id for ssd in o.ssds],
                         [ssd.id for ssd in self.ssds])
        self.assertEqual([owl.id for owl in o.ontologies],
                         [self.ontology.id])
        self.assertEqual(o.name, "octopus-test")
        self.assertEqual(o.modeling_props["compatibleProperties"], True)
        self.assertEqual(o.semantic_type_map["name"],
                         "http://www.semanticweb.org/serene/report_example_ontology#")
        self.assertEqual(o.state,
                         ModelState({
                            "status": "untrained",
                            "message": "",
                            "dateChanged": "2017-03-31T14:09:52.003"
                         }))
        self.assertEqual(o.description, "Testing example for places and companies")

    def test_add(self):
        """
        Tests that the add method functions correctly
        :return:
        """
        server_octo = self._octopus()

        self.assertEqual(server_octo.stored, True)

        # upload a test dataset...
        ds = self._serene.datasets
        data_path = os.path.join("tests", "resources", "data")
        postal_code_ds = ds.upload(os.path.join(data_path, 'postalCodeLookup.csv'))

        # upload a test ssd
        postal_code_ssd = (self._serene
                           .SSD(postal_code_ds, self.ontology, name='postal-code')
                           .map(Column("zipcode"), "Place.postalCode")
                           .map(Column("city"), "City.name")
                           .map(Column("state"), "State.name"))

        # check there is a new ssd available
        ssd_len = len(server_octo.ssds)
        server_octo.add(postal_code_ssd)
        self.assertEqual(ssd_len + 1, len(server_octo.ssds))
        self.assertEqual(server_octo.stored, False)

        # upload a test ontology
        test_ontology = self._ontology()

        # check that there is a new ontology available...
        onto_len = len(server_octo.ontologies)
        server_octo.add(test_ontology)
        self.assertEqual(onto_len + 1, len(server_octo.ontologies))
        self.assertEqual(server_octo.stored, False)

        with self.assertRaises(ValueError):
            server_octo.add("junk")

    def test_remove(self):
        """
        Tests that the add method functions correctly
        :return:
        """
        server_octo = self._octopus()
        self.assertEqual(server_octo.stored, True)

        # record the original lengths
        onto_len = len(server_octo.ontologies)
        ssd_len = len(server_octo.ssds)

        # delete an element...
        server_octo.remove(server_octo.ontologies[0])
        server_octo.remove(server_octo.ssds[0])

        self.assertEqual(onto_len - 1, len(server_octo.ontologies))
        self.assertEqual(ssd_len - 1, len(server_octo.ssds))
        self.assertEqual(server_octo.stored, False)

        with self.assertRaises(ValueError):
            server_octo.remove("something")


    # def test_alignment(self):
    #     """
    #     Tests that the update is passing values correctly...
    #     :return:
    #     """
    #
    #     server_octo = self._octopus()
    #     print(server_octo.train())
    #
    #     if server_octo.state.status in {Status.ERROR}:
    #         print("Something went wrong. Failed to train the Octopus.")
    #         self.fail()
    #
    #     print(server_octo.get_alignment())
    #
    #     self.fail()