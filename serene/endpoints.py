"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Endpoint objects for the user to view and manipulate. These wrap around
server Session objects and call methods to talk to the server.
"""
import os
import json
import pandas as pd
import tempfile
from .utils import gen_id

from functools import lru_cache

from .matcher.dataset import DataSet
from .semantics.ontology import Ontology
from .semantics.ssd import SSDInternal


def decache(func):
    """
    Decorator for clearing the cache. Here we explicitly mark the
    caches that need clearing. There may be a more elegant way to
    do this by having a new lru_cache wrapper that adds the functions
    to a global store, and the cache busters simply clear from this
    list.
    """
    def wrapper(self, *args, **kwargs):
        """
        Wrapper function that busts the cache for each lru_cache file
        """
        if not issubclass(type(self), IdentifiableEndpoint):
            raise ValueError("Can only clear cache of DataSetEndpoint")

        type(self).items.fget.cache_clear()

        return func(self, *args, **kwargs)
    return wrapper


class IdentifiableEndpoint(object):
    """
    An endpoint object that can view and manipulate objects on the server.
    Each object must have a key to identify.
    """
    def __init__(self):
        """
        Only a base type needs to be specified on init, which has to
        contain an `id` variable of type int
        """
        # this is the type of the stored objects
        self._base_type = None

    def _apply(self, func, value, func_name=None):
        """
        Helper function to call `func` with parameter `value` which
        can be an object or an integer key

        :param func: The function to be called
        :param value: An int key or the local object
        :param func_name: A string to use for debugging (optional)
        :return:
        """
        if type(value) == int:
            return func(value)
        elif issubclass(type(value), self._base_type):
            return func(value.id)
        else:
            if func_name is None:
                msg = "Illegal type found: {}".format(type(value))
            else:
                msg = "Illegal type found in {}: {}".format(func_name, type(value))
            raise TypeError(msg)

    @property
    def items(self):
        return tuple()


class DataSetEndpoint(IdentifiableEndpoint):
    """

    :param object:
    :return:
    """
    def __init__(self, session):
        """

        :param self:
        :param api:
        :return:
        """
        super().__init__()
        self._api = session.dataset
        self._base_type = DataSet

    @decache
    def upload(self, filename, description=None, type_map=None):
        """

        :param filename:
        :param description:
        :param type_map:
        :return:
        """
        if issubclass(type(filename), pd.DataFrame):
            path = os.path.join(tempfile.gettempdir(), gen_id() + ".csv")
            df = filename
            df.to_csv(path, index=False)
            filename = path

        assert(issubclass(type(filename), str))

        if not os.path.exists(filename):
            raise ValueError("No filename given.")

        json = self._api.post(
            file_path=filename,
            description=description if description is not None else '',
            type_map=type_map if type_map is not None else {}
        )
        return DataSet(json)

    @decache
    def remove(self, dataset):
        """

        :param dataset:
        :return:
        """
        self._apply(self._api.delete, dataset, 'delete')

    def show(self):
        """
        Prints the datasetlist
        :return:
        """
        print(self.items)

    def get(self, key):
        """Get a single dataset at position key"""
        return DataSet(self._api.item(key))

    @property
    @lru_cache(maxsize=32)
    def items(self):
        """Maintains a list of DataSet objects"""
        keys = self._api.keys()
        ds = []
        for k in keys:
            ds.append(DataSet(self._api.item(k)))
        return tuple(ds)


class OntologyEndpoint(IdentifiableEndpoint):
    """
    User facing object used to control the Ontology endpoint.
    Here the user can view the ontology items, upload an
    ontology, update it etc.

    :param IdentifiableEndpoint: An endpoint with a key value
    :return:
    """
    def __init__(self, session):
        """

        :param session:
        :return:
        """
        super().__init__()
        self._api = session.ontology
        self._base_type = Ontology

    @decache
    def upload(self, ontology, description=None, owl_format=None):
        """
        Uploads an ontology to the Serene server.

        :param ontology:
        :param description:
        :param owl_format:
        :return:
        """
        if issubclass(type(ontology), str):
            # must be a direct filename...
            if not os.path.exists(ontology):
                raise ValueError("No filename given.")
            filename = ontology
            output = Ontology(filename)
        elif issubclass(type(ontology), Ontology):
            # this will use the default path and return it if successful...
            filename = ontology.to_turtle()
            output = ontology
        else:
            raise ValueError("Upload requires Ontology type or direct filename")

        json = self._api.post(
            file_path=filename,
            description=description if description is not None else '',
            owl_format=owl_format if owl_format is not None else 'owl'
        )
        return output.update(json)

    @decache
    def update(self, ontology, file=None, description=None, owl_format=None):
        """
        Uploads an ontology to the Serene server.

        :param ontology:
        :param file:
        :param description:
        :param owl_format:
        :return:
        """
        if not issubclass(type(ontology), Ontology):
            raise ValueError("Update requires Ontology type or direct filename")

        if file is not None:
            # this means we are re-doing the whole thing...
            filename = file
            output = Ontology(filename)
        else:
            # this will use the default path and return it if successful...
            filename = ontology.to_turtle()
            output = ontology

        json = self._api.update(
            file_path=filename,
            description=description,
            owl_format=owl_format
        )
        return output.update(json)

    @decache
    def remove(self, ontology):
        """

        :param ontology:
        :return:
        """
        self._apply(self._api.delete, ontology, 'delete')

    def show(self):
        """
        Prints the ontologylist
        :return:
        """
        print(self.items)

    @property
    @lru_cache(maxsize=32)
    def items(self):
        """Maintains a list of Ontology objects"""
        keys = self._api.keys()
        ontologies = []
        for k in keys:
            on = Ontology(file=self._api.owl_file(k))
            on.update(self._api.item(k))
            ontologies.append(on)
        return tuple(ontologies)


class SSDEndpoint(IdentifiableEndpoint):
    """

    :param object:
    :return:
    """
    def __init__(self, session):
        """

        :param self:
        :param api:
        :return:
        """
        super().__init__()
        self._api = session.ssd
        self._session = session
        self._base_type = SSDInternal

    @decache
    def upload(self, ssd):
        """

        :param ssd
        :return:
        """
        assert(issubclass(type(ssd), SSDInternal))

        if not os.path.exists(ssd.path):
            raise ValueError("No SSD path exists for {}.".format(ssd))

        with open(ssd.path) as f:
            data = json.load(f)

        response = self._api.post(data)

        return ssd.update(response)

    @decache
    def remove(self, ssd):
        """

        :param ssd:
        :return:
        """
        self._apply(self._api.delete, ssd, 'delete')

    def show(self):
        """
        Prints the ssd
        :return:
        """
        print(self.items)

    @property
    @lru_cache(maxsize=32)
    def items(self):
        """Maintains a list of SSD objects"""
        keys = self._api.keys()
        ssd = []
        for k in keys:
            blob = self._api.item(k)
            s = SSDInternal.from_json(blob,
                                      self._session.datasets,
                                      self._session.ontologies)
            ssd.append(s)
        return tuple(ssd)
