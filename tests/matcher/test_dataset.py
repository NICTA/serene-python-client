"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
from serene.elements import DataSet
from io import StringIO
import sys

class TestDataSet(unittest.TestCase):
    """
    Tests the DataSet class
    """
    dataset_json = {
        'dateCreated': '2017-03-16T15:29:03.388',
        'dateModified': '2017-03-16T15:29:03.388',
        'description': '',
        'filename': 'businessInfo.csv',
        'id': 2035625835,
        'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
        'typeMap': {},
        'columns': [
            {
                'datasetID': 2035625835,
                'id': 1246005714,
                'index': 0,
                'logicalType': 'string',
                'name': 'company',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
                'sample': ['Data61'],
                'size': 59
            },
            {
                'datasetID': 2035625835,
                'id': 281689915,
                'index': 1,
                'logicalType': 'string',
                'name': 'ceo',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
                'sample': ['Garv Mcowen'],
                'size': 59
            }
        ]
    }

    equal_dataset_json = {
        'dateCreated': '2017-03-16T15:29:03.388',
        'dateModified': '2017-03-16T15:29:03.388',
        'description': '',
        'filename': 'businessInfo.csv',
        'id': 2035625835,
        'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
        'typeMap': {},
        'columns': [
            {
                'datasetID': 2035625835,
                'id': 1234567890,
                'index': 0,
                'logicalType': 'string',
                'name': 'company',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo2.csv',
                'sample': ['Data61'],
                'size': 59
            },
            {
                'datasetID': 2035625835,
                'id': 123456789,
                'index': 1,
                'logicalType': 'string',
                'name': 'ceo',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo2.csv',
                'sample': ['Garv Mcowen'],
                'size': 59
            }
        ]
    }

    dataset_with_duplicate_column_names_json = {
        'dateCreated': '2017-03-16T15:29:03.388',
        'dateModified': '2017-03-16T15:29:03.388',
        'description': '',
        'filename': 'businessInfo.csv',
        'id': 2035625835,
        'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
        'typeMap': {},
        'columns': [
            {
                'datasetID': 2035625835,
                'id': 1246005714,
                'index': 0,
                'logicalType': 'string',
                'name': 'ceo',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
                'sample': ['Data61'],
                'size': 59
            },
            {
                'datasetID': 2035625835,
                'id': 281689915,
                'index': 1,
                'logicalType': 'string',
                'name': 'ceo',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
                'sample': ['Garv Mcowen'],
                'size': 59
            }
        ]
    }

    def setUp(self):
        self.dataset = DataSet(self.dataset_json)

    def test_stored(self):
        self.assertTrue(self.dataset.stored)

    def test_getitem(self):
        self.assertEqual(self.dataset[1].id, 281689915)

    def test_iter(self):
        self.assertSequenceEqual(
            [column.id for column in self.dataset],
            [1246005714, 281689915])

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_eq(self):
        self.assertEqual(self.dataset, DataSet(self.equal_dataset_json))

    def test_column_names(self):
        self.assertSequenceEqual(self.dataset.column_names(), ["company", "ceo"])

    def test_column(self):
        self.assertEqual(self.dataset.column("ceo").id, 281689915)

    def test_column_with_duplicate_names(self):
        dataset_with_duplicate_column_names = DataSet(
            self.dataset_with_duplicate_column_names_json)
        self.assertRaisesRegex(
            ValueError,
            "Column name ceo is ambiguous.",
            dataset_with_duplicate_column_names.column,
            "ceo")

    def test_column_with_missing_name(self):
        self.assertRaisesRegex(
            ValueError,
            "Column name address does not exist.",
            self.dataset.column,
            "address")

    def test_summary(self):
        output = StringIO()
        try:
            sys_stdout = sys.stdout
            sys.stdout = output
            self.dataset.summary()
        finally:
            sys.stdout = sys_stdout

        self.assertTrue(len(output.getvalue()) > 0)

    def test_repr(self):
        self.assertEqual(
            repr(self.dataset),
            "DataSet(2035625835, businessInfo.csv)")

class TestDataSetList(unittest.TestCase):
    """
    Tests the DataSetList class
    """
    def test_junk(self):
        raise NotImplementedError("Test not implemented")


class TestColumn(unittest.TestCase):
    """
    Tests the DataSet class
    """

    def test_junk(self):
        raise NotImplementedError("Test not implemented")
