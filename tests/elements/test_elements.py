import unittest2 as unittest

from serene.elements.elements import (Class, ClassInstanceLink,
                                      ClassNode, Column, ColumnLink,
                                      DataLink, DataNode,
                                      DataProperty, Mapping,
                                      ObjectLink, ObjectProperty,
                                      ObjectPropertyList, SSDLink)


class TestMapping(unittest.TestCase):
    def setUp(self):
        self.data_property = DataProperty("cores")
        self.column = Column("num_cores")
        self.mapping = Mapping(self.column, self.data_property, predicted=True)

    def test_repr(self):
        self.assertRegex(repr(self.mapping), ".+ -> .+ \[\*\]")


class TestClass(unittest.TestCase):
    def test_ssd_output(self):
        label = "Person"
        prefix = "http://namespace"
        class_instance = Class(label, nodes=["name", "age"], prefix=prefix)
        self.assertEqual(
            class_instance.ssd_output(1),
            {
                "id": 1,
                "label": label,
                "prefix": prefix,
                "type": "Class"
            })

    def test_ssd_output_without_prefix(self):
        label = "Person"
        class_instance = Class(label, nodes=["name", "age"])

        def ssd_output():
            class_instance.ssd_output(1)

        self.assertRaisesRegex(Exception, ".*no prefix.*", ssd_output)

    def test_ssd_output_with_idx(self):
        label = "Person"
        prefix = "http://namespace"
        class_instance = Class(
            label, nodes=["name", "age"], prefix=prefix, idx=1)
        self.assertEqual(
            class_instance.ssd_output(2),
            {
                "id": 1,
                "label": label,
                "prefix": prefix,
                "type": "Class"
            })

    def test_repr(self):
        label = "Person"
        prefix = "http://namespace"
        class_instance = Class(label, nodes=["name", "age"], prefix=prefix)

        self.assertRegex(
            repr(class_instance),
            r"Class\({label}, \[.*name.*, .*age.*\]\)".format(label=label))

    def test_eq(self):
        label = "Person"
        prefix = "http://namespace"
        class_instance1 = Class(label, nodes=["name", "age"], prefix=prefix)
        class_instance2 = Class(label, nodes=["gender"], prefix=prefix)

        self.assertEqual(class_instance1, class_instance2)

    def test_hash(self):
        label = "Person"
        prefix = "http://namespace"
        class_instance = Class(label, nodes=["name", "age"], prefix=prefix)

        self.assertEqual(hash(class_instance), hash((label, prefix)))


class TestDataProperty(unittest.TestCase):
    def setUp(self):
        self.person_name = DataProperty("Person", "name")

    def test_ssd_output(self):
        id = 1

        self.assertEqual(
            self.person_name.ssd_output(id),
            {
                "id": id,
                "label": "Person.name",
                "type": "DataProperty"
            })

    def test_ne(self):
        data_property = DataProperty("Person", "age")

        self.assertNotEqual(data_property, self.person_name)

    def test_eq_with_no_class(self):
        name = DataProperty("name")

        self.assertEqual(self.person_name, name)

    def test_eq(self):
        data_property = DataProperty("Person", "name")

        self.assertEqual(self.person_name, data_property)

    def test_repr(self):
        self.assertEqual(repr(self.person_name), "DataProperty(Person, name)")

    def test_hash(self):
        self.assertEqual(hash(self.person_name), hash(("Person", "name")))


class TestObjectProperty(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        dst = Class("Person", nodes=["name"], prefix=self.prefix)
        self.obj_prop = ObjectProperty(
            "has-a", dst=dst, src=dst.nodes[0], prefix=self.prefix)

    def test_ssd_output(self):
        index = 1
        source_index = 2
        target_index = 3
        index_map = {
            self.obj_prop.src: source_index,
            self.obj_prop.dst: target_index
        }

        self.assertEqual(
            self.obj_prop.ssd_output(index, index_map),
            {
                "id": index,
                "source": source_index,
                "target": target_index,
                "label": self.obj_prop.label,
                "type": self.obj_prop.link_type,
                "prefix": self.obj_prop.prefix
            }
        )

    def test_repr(self):
        pass

    def test_eq(self):
        dst = Class("Person", nodes=["name"], prefix=self.prefix)
        obj_prop = ObjectProperty(
            "has-a", dst=dst, src=dst.nodes[0], prefix=self.prefix)

        self.assertEqual(self.obj_prop, obj_prop)

    def test_ne(self):
        dst = Class("Person", nodes=["age"], prefix=self.prefix)
        obj_prop = ObjectProperty(
            "has-a", dst=dst, src=dst.nodes[0], prefix=self.prefix)

        self.assertNotEqual(self.obj_prop, obj_prop)

    def test_hash(self):
        self.assertEqual(
            hash(self.obj_prop),
            hash((self.obj_prop.src, self.obj_prop.dst, self.obj_prop.label)))


class TestObjectPropertyList(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        dst = Class("Person", nodes=["name"], prefix=self.prefix)
        self.prop = ObjectProperty(
            "has-a", dst=dst, src=dst.nodes[0], prefix=self.prefix)

        self.props = ObjectPropertyList(self.prop)

    def test_len(self):
        self.assertEqual(len(self.props), 1)

    def test_getitem(self):
        self.assertEqual(self.props[0], self.prop)

    def test_delitem(self):
        del self.props[0]

        self.assertEqual(len(self.props), 0)

    def test_setitem(self):
        dst = Class("Dog", nodes=["height"], prefix=self.prefix)
        prop = ObjectProperty(
            "has-a", dst=dst, src=dst.nodes[0], prefix=self.prefix)
        self.props[0] = prop

        self.assertEqual(self.props[0], prop)

    def test_insert(self):
        dst = Class("Dog", nodes=["height"], prefix=self.prefix)
        prop = ObjectProperty(
            "has-a", dst=dst, src=dst.nodes[0], prefix=self.prefix)
        self.props.insert(0, prop)

        self.assertSequenceEqual(self.props, [prop, self.props[1]])

    def test_check(self):
        def check():
            ObjectPropertyList.check(1)

        self.assertRaisesRegex(TypeError, ".*ObjectProperty.*permitted", check)

    def test_repr(self):
        self.props.insert(0, self.prop)

        self.assertRegex(repr(self.props), ".*\n.*")


class TestClassNode(unittest.TestCase):
    def setUp(self):
        self.label = "Person"
        self.index = 1
        self.prefix = "http://namespace"
        self.node = ClassNode(self.label, self.index, self.prefix)

    def test_full_label(self):
        self.assertEqual(self.node.full_label, self.label)

    def test_label(self):
        self.assertEqual(self.node.full_label, self.label)

    def test_index(self):
        self.assertEqual(self.node.index, self.index)

    def test_prefix(self):
        self.assertEqual(self.node.prefix, self.prefix)

    def test_type(self):
        self.assertEqual(self.node.type, "ClassNode")

    def test_repr(self):
        self.assertEqual(
            repr(self.node),
            "ClassNode({}, {}, {})".format(
                self.label, self.prefix, self.index))

    def test_ne(self):
        self.assertNotEqual(self.node, ClassNode(self.label, 2, self.prefix))

    def test_eq(self):
        self.assertEqual(
            self.node, ClassNode(self.label, self.index, self.prefix))

    def test_hash(self):
        self.assertEqual(
            hash(self.node),
            hash(
                (self.index, self.label, str(self.prefix), type(self.node))
            )
        )


class TestDataNode(unittest.TestCase):
    def setUp(self):
        self.class_label = "Person"
        self.class_index = 1
        self.prefix = "http://namespace"
        self.class_node = ClassNode(
            self.class_label, self.class_index, self.prefix)

        self.data_label = "name"
        self.data_index = 11
        self.data_node = DataNode(
            self.class_node,
            self.data_label,
            index=self.data_index,
            prefix=self.prefix)

    def test_label(self):
        self.assertEqual(self.data_node.label, self.data_label)

    def test_class_node(self):
        self.assertEqual(self.data_node.class_node, self.class_node)

    def test_index(self):
        self.assertEqual(self.data_node.index, self.data_index)

    def test_prefix(self):
        self.assertEqual(self.data_node.prefix, self.prefix)

    def test_type(self):
        self.assertEqual(self.data_node.type, "DataNode")

    def test_dtype(self):
        self.assertEqual(self.data_node.dtype, str)

    def test_full_label(self):
        self.assertEqual(
            self.data_node.full_label,
            "{}.{}".format(self.class_label, self.data_label))

    def test_repr(self):
        self.assertEqual(
            repr(self.data_node),
            "DataNode({}, {}, {})".format(
                self.class_node,
                self.data_label,
                self.data_index
            )
        )

    def test_ne(self):
        data_node = DataNode(
            self.class_node,
            "age",
            index=12,
            prefix=self.prefix)

        self.assertNotEqual(self.data_node, data_node)

    def test_eq(self):
        data_node = DataNode(
            self.class_node,
            self.data_label,
            index=self.data_index,
            prefix=self.prefix)

        self.assertEqual(self.data_node, data_node)

    def test_hash(self):
        self.assertEqual(
            hash(self.data_node),
            hash((
                self.data_label,
                str(self.prefix),
                self.class_node,
                self.data_index
            ))
        )


class TestSSDLink(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        self.label = "link1"
        self.link = SSDLink(self.label, self.prefix)

    def test_ne(self):
        self.assertNotEqual(
            self.link,
            SSDLink("link2", self.prefix))

    def test_eq(self):
        self.assertEqual(
            self.link,
            SSDLink(self.label, self.prefix))

    def test_hash(self):
        self.assertEqual(
            hash(self.link),
            hash((self.label, str(self.prefix))))


class TestDataLink(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        self.label = "link1"
        self.link = DataLink(self.label, self.prefix)

    def test_repr(self):
        self.assertEqual(
            repr(self.link),
            "DataLink({})".format(self.label))


class TestObjectLink(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        self.label = "link1"
        self.link = ObjectLink(self.label, self.prefix)

    def test_repr(self):
        self.assertEqual(
            repr(self.link),
            "ObjectLink({})".format(self.label))


class TestColumnLink(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        self.label = "link1"
        self.link = ColumnLink(self.label, self.prefix)

    def test_repr(self):
        self.assertEqual(
            repr(self.link),
            "ColumnLink({}, {})".format(self.label, self.prefix))

    def test_eq(self):
        self.assertEqual(
            self.link,
            ColumnLink(self.label, self.prefix))


class TestClassInstanceLink(unittest.TestCase):
    def setUp(self):
        self.prefix = "http://namespace"
        self.label = "link1"
        self.link = ClassInstanceLink(self.label, self.prefix)

    def test_repr(self):
        self.assertEqual(
            repr(self.link),
            "ClassLink({})".format(self.label))
