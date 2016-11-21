import unittest2 as unittest
import dataint


class TestSemanticModeller(unittest.TestCase):
    """
    Tests the SemanticSourceDesc
    """
    def test_junk(self):
        sm = dataint.SemanticModeller(
            config={
                "kmeans": 5,
                "threshold": 0.7,
                "search-depth": 100
            },
            ontologies=[
                "/some/file/something.owl",
                "/some/other/file/something.owl"
            ]
        )
        self.assertEqual('a', 'a')

    def test_init(self):
        pass

    def test_add_ontology(self):
        pass

    def test_to_ssd(self):
        pass

    def test_flatten(self):
        pass

    def test_class_nodes(self):
        pass

    def test_data_nodes(self):
        pass

    def test_links(self):
        pass

    def test_ontologies(self):
        pass


class TestSemanticSourceDesc(unittest.TestCase):
    """
    Testing the SemanticSourceDesc object...
    """

    def test_init__(self):
        pass

    def test_find(self):
        pass

    def test_map(self):
        pass

    def test_link(self):
        pass

    def test_sample(self):
        pass

    def test_remove(self):
        pass

    def test_predict(self):
        pass

    def test_show(self):
        pass

    def test_save(self):
        pass

    def test_version(self):
        pass

    def test_ontologies(self):
        pass

    def test_ssd(self):
        pass

    def test_json(self):
        pass
    
    def test_model(self):
        pass

    @property
    def predictions(self):
        """The list of all predicted mappings"""
        return list(m for m in self.mappings if m.predicted)


    @property
    def mappings(self):
        """The list of all current mappings"""
        return list(self._mapping.values())


    @property
    def columns(self):
        """The list of columns currently used"""
        return list(self._mapping.keys())


    @property
    def transforms(self):
        """All available transforms"""
        return list(self._transforms)


    @property
    def data_nodes(self):
        """All available data nodes"""
        return list(m.node for m in self.mappings)


    @property
    def class_nodes(self):
        """All available class nodes"""
        return list(set(n.parent for n in self.data_nodes if n is not None))


    @property
    def links(self):
        """
            The links in the SSD. The links come from 2 sources. The object links come from
            what's defined in the semantic model. The data links come
            from the relevant mappings to the columns. The other links
            are not necessary.
        """
        # grab the object links from the model...
        object_links = [link for link in self._model.links
                        if link.link_type == Link.OBJECT_LINK]

        # grab the data links from the model...
        data_links = [link for link in self._model.links
                      if link.link_type == Link.DATA_LINK]

        # combine the relevant links...
        return object_links + [link for link in data_links if link.dst in self.data_nodes]


    def __repr__(self):
        """
        Displays the maps of columns to the data nodes of the Ontology
        :return:
        """
        map_str = [str(m) for m in self.mappings]
        link_str = [str(link) for link in self.links if link.link_type == Link.OBJECT_LINK]

        items = map_str + link_str
        full_str = '\n\t'.join(items)

        return "[\n\t{}\n]".format(full_str)
