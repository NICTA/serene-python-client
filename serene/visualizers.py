import os.path
import tempfile
import webbrowser

import pygraphviz as pgv

from .elements import IdentTransform, Link


class BaseVisualizer(object):
    """
    Basic Visualizer object for drawing a BaseSemantic object
    """

    def __init__(self, struct, outfile=None):
        self.struct = struct
        if outfile is None:
            self.outfile = os.path.join(tempfile.gettempdir(), 'out.png')
        else:
            self.outfile = outfile

    def _draw_class_nodes(self, graph):
        """
        Draws the ClassNode objects onto the pygraphviz graph.

        Here we just show the model class nodes and links...

        :param graph:
        :return:
        """
        for c in self.struct.class_nodes:
            graph.add_node(c,
                           label=c.name,
                           color='white',
                           style='filled',
                           fillcolor='#59d0a0',
                           shape='ellipse',
                           fontname='helvetica',
                           rank='source')

        for c in self.struct.class_nodes:
            if c.parent is not None:
                graph.add_edge(c,
                               c.parent,
                               label='subclass',
                               color='#59d0a0',
                               style='dashed',
                               fontname='helvetica')

                # ensure that the parent is in the graph!
                graph.add_node(c.parent,
                               label=c.parent.name,
                               color='white',
                               style='filled',
                               fillcolor='#59d0a0',
                               shape='ellipse',
                               fontname='helvetica',
                               rank='source')

        for link in self.struct.links:
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

        for d in self.struct.data_nodes:
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

    def _draw_elements(self, g):
        """

        :param g:
        :return:
        """
        self._draw_class_nodes(g)
        self._draw_data_nodes(g)

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

        self._draw_elements(g)

        g.draw(self.outfile, prog='dot')
        webbrowser.open("file://{}".format(os.path.abspath(self.outfile)))


class SSDVisualizer(BaseVisualizer):
    """
    Visualizer object for drawing an SSD object
    """

    def __init__(self, struct, outfile=None):
        super().__init__(struct, outfile)

    def _draw_columns(self, graph):
        """
        Draws the column objects onto the pygraphviz graph
        :param graph:
        :return:
        """
        for col in self.struct.columns:
            graph.add_node(col,
                           style='filled',
                           label=col.name,
                           color='white',
                           shape='box',
                           fontname='helvetica')

        graph.add_subgraph(self.struct.columns,
                           rank='max',
                           name='cluster1',
                           color='#f09090',
                           fontcolor='#c06060',
                           label='source',
                           style='filled',
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
                 for n in self.struct.mappings
                 if n is not None]

        for d in nodes:
            if d is not None:
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
        for m in self.struct.mappings:
            if m.node is not None:
                graph.add_edge(m.node,
                               m.column,
                               color='brown',
                               fontname='helvetica-italic',
                               style='dashed')
                # graph.add_edge(m.node,
                #                m.transform,
                #                color='brown',
                #                fontname='helvetica-italic',
                #                style='dashed')

    def _draw_elements(self, g):
        """
        Here we override the base class, to draw all the
        additional features...

        :param g:
        :return:
        """
        self._draw_class_nodes(g)
        self._draw_data_nodes(g)
        self._draw_columns(g)
        #self._draw_transforms(g)
        self._draw_mappings(g)
        return
