import json
import os.path
import tempfile
import webbrowser
from collections import OrderedDict

import pygraphviz as pgv

from .elements import IdentTransform
from .semantics import Link


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


class SSDVisualizer(object):
    """
    Visualizer object for drawing an SSD object
    """

    def __init__(self, ssd, outfile=None):
        self.ssd = ssd
        if outfile is None:
            self.outfile = os.path.join(tempfile.gettempdir(), 'out.png')
        else:
            self.outfile = outfile

    def _draw_columns(self, graph):
        """
        Draws the column objects onto the pygraphviz graph
        :param graph:
        :return:
        """
        for col in self.ssd.columns:
            graph.add_node(col,
                           style='filled',
                           label=col.name,
                           color='white',
                           shape='box',
                           fontname='helvetica')

        graph.add_subgraph(self.ssd.columns,
                           rank='max',
                           name='cluster1',
                           color='#f09090',
                           fontcolor='#c06060',
                           label='source',
                           style='filled',
                           fontname='helvetica')

    def _draw_transforms(self, graph):
        """
        Draws the Transform objects onto the pygraphviz graph
        :param graph:
        :return:
        """
        for t in self.ssd.transforms:
            if type(t) == IdentTransform:
                transform_text = "Identity".format(t.id)
            else:
                transform_text = "Transform({})".format(t.id)

            graph.add_node(t,
                           label=transform_text,
                           color='white',
                           style='filled',
                           shape='box',
                           fontname='helvetica')

        graph.add_subgraph(self.ssd.transforms,
                           rank='min',
                           name='cluster2',
                           style='filled',
                           color='#9090f0',
                           fontcolor='#6060c0',
                           label='transforms',
                           fontname='helvetica')

    def _draw_class_nodes(self, graph):
        """
        Draws the ClassNode objects onto the pygraphviz graph.

        Here we just show the model class nodes and links...

        :param graph:
        :return:
        """
        for c in self.ssd.class_nodes:
            graph.add_node(c,
                           label=c.name,
                           color='white',
                           style='filled',
                           fillcolor='#59d0a0',
                           shape='ellipse',
                           fontname='helvetica',
                           rank='source')

        for link in self.ssd.links:
            if link.link_type == Link.OBJECT_LINK:
                graph.add_edge(link.src,
                               link.dst,
                               label=link.name,
                               color='#59d0a0',
                               fontname='helvetica')

    def _draw_data_nodes(self, graph):
        """
        Draws the DataNode objects onto the pygraphviz graph

        Note that we don't want all the data nodes, just the
        ones that have been mapped to columns...

        :param graph:
        :return:
        """

        nodes = [n.node
                 for n in self.ssd.mappings
                 if n is not None]

        for d in nodes:
            graph.add_node(d,
                           label=d.name,
                           color='white',
                           style='filled',
                           fontcolor='black',
                           shape='ellipse',
                           fontname='helvetica',
                           rank='same')

            if d.parent is not None:
                graph.add_edge(d.parent,
                               d,
                               color='gray',
                               fontname='helvetica-italic')

        graph.add_subgraph(nodes,
                           rank='same',
                           name='cluster3',
                           style='filled',
                           color='#e0e0e0',
                           fontcolor='#909090',
                           fontname='helvetica')

    def _draw_mappings(self, graph):
        """
        Draws the mappings from columns to transforms to DataNodes
        in pyGraphViz

        :param g:
        :return:
        """
        for m in self.ssd.mappings:
            if m.node is not None:
                graph.add_edge(m.transform,
                               m.column,
                               color='brown',
                               fontname='helvetica-italic',
                               style='dashed')
                graph.add_edge(m.node,
                               m.transform,
                               color='brown',
                               fontname='helvetica-italic',
                               style='dashed')

    def show(self):
        """
        Show the graph using pygraphviz
        :return:
        """
        g = pgv.AGraph(strict=False,
                       directed=True,
                       remincross='true',
                       overlap=False,
                       splines='true',
                       fontname='helvetica')

        self._draw_class_nodes(g)
        self._draw_data_nodes(g)
        self._draw_columns(g)
        self._draw_transforms(g)
        self._draw_mappings(g)

        g.draw(self.outfile, prog='dot')
        webbrowser.open("file://{}".format(os.path.abspath(self.outfile)))