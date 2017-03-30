from urllib.parse import urljoin

from requests import Response, Session
from unittest2 import TestCase

from mock import Mock
from serene.api.data_api import DataSetAPI
from serene.api.exceptions import InternalError
from functools import partial


class TestDataSetAPI(TestCase):
    def setUp(self):
        self.connection = Mock(Session())
        self.root_uri = "http://localhost/"
        self.dataset_path = "dataset/"
        self.uri = self.root_uri + self.dataset_path
        self.api = DataSetAPI(self.root_uri, self.connection)
        self.response = Mock(Response())

        self.description = "this python file"
        self.file_path = __file__
        self.type_map = {"a": "int"}

        self.update_description = "another python file"
        self.update_type_map = {"b": "string"}

    def test_keys(self):
        keys = [1, 2]
        self.response.status_code = 200
        self.response.json = Mock(return_value=keys)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.keys()

        self.assertEqual(result, keys)
        self.connection.get.assert_called_with(self.uri)

    def test_keys_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)

        self.assertRaises(InternalError, self.api.keys)

    def test_post(self):
        message = "Created"
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.post(self.description, self.file_path, self.type_map)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri)
        self.assertEqual(
            args[1]["data"],
            {"description": self.description, "typeMap": self.type_map})
        self.assertIsNotNone(args[1]["files"])
        self.assertEqual(result, message)

    def test_post_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_post = partial(
            self.api.post, self.description, self.file_path, self.type_map)

        self.assertRaises(InternalError, api_post)

    def test_update(self):
        message = "Updated"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.update(
            key, self.update_description, self.update_type_map)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri + str(key))
        self.assertEqual(
            args[1]["data"],
            {
                "description": self.update_description,
                "typeMap": self.update_type_map
            })
        self.assertEqual(result, message)

    def test_update_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_update = partial(
            self.api.update, 1, self.update_description, self.update_type_map)

        self.assertRaises(InternalError, api_update)

    def test_item(self):
        key = 1
        item = {
            "description": self.description,
            "typeMap": self.type_map
        }
        self.response.status_code = 200
        self.response.json = Mock(return_value=item)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.item(key)

        self.assertEqual(result, item)
        self.connection.get.assert_called_with(self.uri + str(key))

    def test_item_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)
        api_item = partial(self.api.item, 1)

        self.assertRaises(InternalError, api_item)

    def test_delete(self):
        message = "Deleted"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.delete = Mock(return_value=self.response)
        result = self.api.delete(key)

        self.assertEqual(result, message)
        self.connection.delete.assert_called_with(self.uri + str(key))

    def test_delete_with_connection_exception(self):
        self.connection.delete = Mock(side_effect=Exception)
        api_delete = partial(self.api.delete, 1)

        self.assertRaises(InternalError, api_delete)
