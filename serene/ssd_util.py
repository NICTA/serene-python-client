import json
from collections import OrderedDict


class SSDJsonBuilder(object):
    """
    Helper class to build up the json output for the SSD file.
    """
    def __init__(self, ssd):
        """

        :param ssd:
        """
        self._ssd = ssd

        self._all_nodes = [n for n in
                           self._ssd.class_nodes + self._ssd.data_nodes
                           if n is not None]

        self._node_map = {m: i for i, m in enumerate(self._all_nodes)}
        self._attr_map = {m: i for i, m in enumerate(self._ssd.mappings)}

    def to_dict(self):
        """

        :return:
        """
        d = OrderedDict()
        d["version"] = self._ssd.version
        d["name"] = self._ssd.file
        d["columns"] = self.columns
        d["attributes"] = self.attributes
        d["ontology"] = [o.filename for o in self._ssd.ontologies]
        d["semanticModel"] = self.semantic_model
        d["mappings"] = self.mappings

        return d

    def to_json(self):
        """

        :return:
        """
        return json.dumps(self.to_dict(), indent=4)

    @property
    def columns(self):
        """

        :return:
        """
        return [
            {
                "id": c.index,
                "name": c.name
            } for c in self._ssd.columns]

    @property
    def attributes(self):
        """

        :return:
        """
        return [
            {
                "id": self._attr_map[m],
                "name": m.column.name,
                "label": m.transform.name,
                "columnIds": [m.column.index],
                "sql": m.transform.apply(m.column)
            } for m in self._ssd.mappings]

    @property
    def semantic_model(self):
        """

        :return:
        """
        return {
            "nodes": [node.ssd_output(index)
                      for index, node in enumerate(self._all_nodes)],
            "links": [link.ssd_output(self._node_map)
                      for link in self._ssd.links]
        }

    @property
    def mappings(self):
        """

        :return:
        """
        return [
            {
                "attribute": self._attr_map[m],
                "node": self._node_map[m.node]
            } for m in self._ssd.mappings if m.node is not None]


