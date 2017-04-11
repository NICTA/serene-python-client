"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import serene
import os
from .utils import TestWithServer


class TestSerene(TestWithServer):
    """
    Tests the SemanticModeller class
    """
    def __init__(self, method_name="runTest"):

        super().__init__(method_name)

        path = os.path.join(os.path.dirname(__file__), "resources")

        self._test_owl = os.path.join(path, 'owl', 'dataintegration_report_ontology.ttl')

        self._business_file = os.path.join(path, 'data', 'businessInfo.csv')
        self._cities_file = os.path.join(path, 'data', 'getCities.csv')
        self._employee_file = os.path.join(path, 'data', 'getEmployees.csv')
        self._postal_file = os.path.join(path, 'data', 'postalCodeLookup.csv')
        self._addresses_file = os.path.join(path, 'data', 'EmployeeAddressesSingleName.csv')

        self._map_file = os.path.join(path, 'data', 'example-map-file.csv')
        self._link_file = os.path.join(path, 'data', 'example-link-file.csv')

    def setUp(self):
        self._clear_storage()

    def tearDown(self):
        self._clear_storage()

    def _clear_storage(self):
        """Removes all server elements"""
        for o in self._serene.octopii.items:
            self._serene.octopii.remove(o)

        for ssd in self._serene.ssds.items:
            self._serene.ssds.remove(ssd)

        for ds in self._serene.datasets.items:
            self._serene.datasets.remove(ds)

        for on in self._serene.ontologies.items:
            self._serene.ontologies.remove(on)

    def test_load(self):
        """
        Tests the loading mechanism
        :return:
        """
        ds, on, ssds = self._serene.load(self._test_owl,
                                         [
                                             self._business_file,
                                             self._cities_file,
                                             self._employee_file,
                                             self._postal_file,
                                             self._addresses_file
                                         ],
                                         self._map_file,
                                         self._link_file)
        self.assertEqual(len(ds), 5)
        self.assertEqual(len(on), 1)
        self.assertEqual(len(ssds), 5)
