"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import sys
from io import StringIO

import unittest2 as unittest

from serene.elements import DataSet
from serene.elements import DataSetList

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

    dataset_json1 = dataset_json = {
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

    dataset_json2 = dataset_json = {
        'dateCreated': '2017-03-16T15:29:03.388',
        'dateModified': '2017-03-16T15:29:03.388',
        'description': '',
        'filename': 'businessInfo2.csv',
        'id': 2035625836,
        'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv',
        'typeMap': {},
        'columns': [
            {
                'datasetID': 2035625836,
                'id': 2246005714,
                'index': 0,
                'logicalType': 'string',
                'name': 'company',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo2.csv',
                'sample': ['Data61'],
                'size': 59
            },
            {
                'datasetID': 2035625836,
                'id': 381689915,
                'index': 1,
                'logicalType': 'string',
                'name': 'ceo',
                'path': '/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo2.csv',
                'sample': ['Garv Mcowen'],
                'size': 59
            }
        ]
    }

    def setUp(self):
        self.dataset1 = DataSet(self.dataset_json1)
        self.dataset2 = DataSet(self.dataset_json2)
        self.datasetList = DataSetList(self.dataset1, self.dataset2)

    def test_check(self):
        self.assertIsNone(DataSetList.check(self.dataset1))

    def test_check_with_non_dataset(self):
        checked = 1
        self.assertRaisesRegex(
            TypeError,
            "Only DataSet types permitted: {}".format(checked),
            DataSetList.check,
            checked)

    def test_len(self):
        self.assertEqual(len(self.datasetList), 2)

    def test_getitem(self):
        self.assertIs(self.datasetList[1], self.dataset2)

    def test_delitem(self):
        def delete(): del self.datasetList[0]
        self.assertRaisesRegex(
            Exception,
            "Use Serene\.remove_dataset to correctly remove dataset",
            delete)

    def test_setitem(self):
        self.datasetList[1] = self.dataset1
        self.assertIs(self.datasetList[1], self.dataset1)

    def test_setitem_with_non_dataset(self):
        def setitem(): self.datasetList[1] = 1
        self.assertRaises(TypeError, setitem)

    def test_insert(self):
        self.datasetList.insert(1, self.dataset1)
        self.assertSequenceEqual(
            self.datasetList,
            [self.dataset1, self.dataset1, self.dataset2])

    def test_insert_with_non_dataset(self):
        def insert(): self.datasetList.insert(1, 1)
        self.assertRaises(TypeError, insert)

    def test_repr(self):
        self.assertEqual(
            repr(self.datasetList),
            "[{}]".format(
                "\n".join([
                    repr(self.dataset1), repr(self.dataset2)
                ])
            )
        )

    def test_summary(self):
        self.assertEqual(self.datasetList.summary.shape[0], 2)

class TestColumn(unittest.TestCase):
    """
    Tests the DataSet class
    """

    def test_junk(self):
        raise NotImplementedError("Test not implemented")
