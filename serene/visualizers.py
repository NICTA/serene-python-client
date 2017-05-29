import os.path
import tempfile
import webbrowser

import pygraphviz as pgv

from .elements import ObjectProperty, ClassNode, DataNode, Column
from .elements import DataLink, ColumnLink, ObjectLink, ClassInstanceLink, SubClassLink


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

    def _draw_elements(self, g):
        """Function to draw all elements onto the graph"""
        pass

    def show(self, title=''):
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

        if title and len(title):
            g.graph_attr['label'] = title

        self._draw_elements(g)

        print()
        print(g)
        print()

        g.draw(self.outfile, prog='dot')
        # g.draw("paper.pdf", prog='dot')
        webbrowser.open("file://{}".format(os.path.abspath(self.outfile)))


class OntologyVisualizer(BaseVisualizer):
    """
    Visualizer object for drawing the Ontology
    """
    def _draw_class_nodes(self, graph):
        """
        Draws the ClassNode objects onto the pygraphviz graph.

        Here we just show the model class nodes and links...

        :param graph:
        :return:
        """
        for c in self.struct.class_nodes:
            graph.add_node(c,
                           label=c.label,
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
                               label=c.parent.label,
                               color='white',
                               style='filled',
                               fillcolor='#59d0a0',
                               shape='ellipse',
                               fontname='helvetica',
                               rank='source')

        for link in self.struct.links:
            if link.link_type == ObjectProperty.OBJECT_LINK:
                graph.add_edge(link.src,
                               link.dst,
                               label=link.label,
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
                           label=d.label,
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
        Draws the class nodes and data nodes
        :param g:
        :return:
        """
        self._draw_class_nodes(g)
        self._draw_data_nodes(g)


class SSDVisualizer(BaseVisualizer):
    """
    Visualizer object for drawing an SSD object
    """
    def __init__(self, struct, outfile=None, name='source'):
        super().__init__(struct, outfile)

        self._name = name

        # the functions used to add nodes for each type
        self.node_functions = {
            ClassNode: self._add_class_node,
            DataNode: self._add_data_node,
            Column: self._add_column_node,
        }

        # the functions used to add links for each type
        self.link_functions = {
            ObjectLink: self._add_object_link,
            DataLink: self._add_data_link,
            ColumnLink: self._add_column,
            ClassInstanceLink: self._add_class_link,
            SubClassLink: self._add_object_link
        }

    def _filter_nodes(self, value_type):
        """Helper function to grab node ids from the graph structure"""
        z = self.struct
        g = z.graph
        return [n for n in g.nodes()
                if issubclass(type(z.node_data(n)), value_type)]

    @staticmethod
    def _add_class_node(graph, key, node):
        """Add the class node to the graph"""
        return graph.add_node(key,
                              label=node.label,
                              color='white',
                              style='filled',
                              fillcolor='#59d0a0',
                              shape='ellipse',
                              fontname='helvetica'
                              # , rank='source'
                              )

    @staticmethod
    def _add_data_node(graph, key, node):
        """Add the data node to the graph"""
        return graph.add_node(key,
                              label=node.label,
                              color='white',
                              style='filled',
                              fontcolor='black',
                              shape='ellipse',
                              fontname='helvetica'
                              # ,rank='same'
                              )

    @staticmethod
    def _add_column_node(graph, key, node):
        """Add the column to the graph"""
        graph.add_node(key,
                       style='filled',
                       label=node.name,
                       color='white',
                       shape='box',
                       fontname='helvetica')

    def _add_data_bounds(self, graph):
        """Add the bounding box for the data nodes"""
        graph.add_subgraph(self._filter_nodes(DataNode),
                           rank='same',
                           name='cluster2',
                           style='filled',
                           color='#e0e0e0',
                           label='',
                           fontcolor='#909090',
                           fontname='helvetica')

    def _add_class_cluster(self, graph):
        """Add subgraph for the class nodes"""
        graph.add_subgraph(self._filter_nodes(ClassNode))

    def _add_column_bounds(self, graph):
        """Add the bounding box for the columns"""
        graph.add_subgraph(self._filter_nodes(Column),
                           rank='same',
                           name='cluster3',
                           color='#f09090',
                           fontcolor='#c06060',
                           label=self._name,
                           style='filled',
                           fontname='helvetica')

    @staticmethod
    def _add_object_link(graph, src, dst, link):
        """Function to add the ObjectLinks"""
        graph.add_edge(src, dst,
                       label=link.label,
                       color='#59d0a0',
                       fontname='helvetica')

    @staticmethod
    def _add_data_link(graph, src, dst, _):
        """Function to add the DataLinks"""
        graph.add_edge(src, dst,
                       color='gray',
                       fontname='helvetica-italic')

    @staticmethod
    def _add_class_link(graph, src, dst, _):
        """Function to add the ClassInstanceLinks.
        For now they are treated as DataPropertyLinks..."""
        graph.add_edge(src, dst,
                       color='gray',
                       fontname='helvetica-italic')

    @staticmethod
    def _add_column(graph, src, dst, _):
        """Function to add the Column objects"""
        graph.add_edge(src, dst,
                       color='brown',
                       fontname='helvetica-italic',
                       style='dashed')

    def _draw_nodes(self, graph):
        """Draw the node objects using the node_functions object"""
        # take the graph core from the SSDGraph
        g = self.struct.graph

        # first add all the nodes
        for key in g.nodes():
            # extract the data objects from all nodes
            node = g.node[key][self.struct.DATA_KEY]
            if type(node) in self.node_functions:
                self.node_functions[type(node)](graph, key, node)
            else:
                msg = "Unknown node type in graph: {}".format(node)
                raise Exception(msg)


    def _draw_bounds(self, graph):
        """Draw the bounding boxes for the DataNodes and Columns"""
        # add the boundaries...
        self._add_class_cluster(graph)
        self._add_data_bounds(graph)
        self._add_column_bounds(graph)

    def _draw_links(self, graph):
        """Draw the link objects using the link_functions object"""
        # take the graph core from the SSDGraph
        g = self.struct.graph

        # add the links
        for src, dst in g.edges():
            for e in g.edge[src][dst].values():
                link = e[self.struct.DATA_KEY]
                if type(link) in self.link_functions:
                    self.link_functions[type(link)](graph, src, dst, link)
                else:
                    msg = "Unknown link type in graph: {}".format(link)
                    raise Exception(msg)

    def _draw_elements(self, graph):
        """
        Using the SSD graph object, we draw all the node types,
        then the bounding boxes and finally the links
        :param graph: The graphviz draw object
        :return: None
        """
        self._draw_nodes(graph)
        self._draw_links(graph)
        self._draw_bounds(graph)
