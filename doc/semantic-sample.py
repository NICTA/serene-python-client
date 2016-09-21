"""
 input
  Some knowledge about the ontology or where you want to end up...

  /some/junk/test1.csv  - contains columns: name, birth, city, state, workplace
  /some/junk/test2.csv  - contains columns: state, city
  /some/junk/test3.csv  - contains columns: company, ceo, city, state
  /some/junk/test4.csv  - contains columns: employer, employee
  /some/junk/test5.csv  - contains columns: zipcode, city, state
  ..
  ..
  ..
  ...

 output
  SSD: semantic source description files
    /some/junk/test1.ssd
    /some/junk/test2.ssd
    /some/junk/test3.ssd
    /some/junk/test4.ssd
    /some/junk/test5.ssd
    ...
    ...
    ...

"""
#===========
# SETUP
#===========

# start with a SemanticModeller object...
sm = dataint.SemanticModeller(
    config={
        "kmeans": 5,
        "threshold": 0.7,
        "search-depth": 100
    },
    ontologies=["/some/file/something.owl", "/something/else/person.owl"] # optional
)
"""
>>> <SemanticModeller 98y3498yrt>
"""

# build up a simple ontology
on = (
  dataint.Ontology()
    .add_prefix("asd", "asd:/9e8rgyusdgfiuhsdf.dfguhudfh")
    .uri("http://www.semanticweb.org/data_integration_project/report_example_ontology")
    .add_class_node("Person", ["name", "address", "phone"], prefix="xsd")
    .add_class_node(
      "Car",
      {
        "make": str,
        "model": str,
        "VIN": int
      })
      .add_relationship("Person", "owns", "Car")
      .add_class_node("Location")
      .add_class_node("City", parent="Location")
      .add_class_node("State", parent="Location")
      .add_relationship("City", "is_in", "State")
  )

# add the ontology to the SemanticModeller
sm.add_ontology(on)
"""
>>> <SemanticModeller 98y3498yrt>
"""

# list all the class nodes
sm.class_nodes
"""
>>>
[
    ClassNode(Person, [name, dob, phone])
    ClassNode(Car, [name, address, phone])
    ClassNode(Event, [location, time, date])
    ClassNode(Location, [lat, long])
    ClassNode(City, [name])
]
"""

# list all the data nodes
sm.data_nodes
"""
>>>
[
    DataNode(name, ClassNode(Person))
    DataNode(dob, ClassNode(Person))
    DataNode(phone, ClassNode(Person))
    DataNode(location, ClassNode(Event))
    DataNode(time, ClassNode(Event))
    DataNode(date, ClassNode(Event))
    ...
]
"""

#=========================
# CREATING SEMANTIC MODELS
#=========================

# once we load the file, the columns exist in the SemanticSourceDescription(SSD),
# but they have no DataNodes assigned...
ssd = sm.load("/some/junk/test1.csv")
ssd
"""
>>>
[
    Column(name, 1) -> None
    Column(birth, 2) -> None
    Column(city, 3) -> None
    Column(state, 4) -> None
    Column(workplace, 5) -> None
]
"""

# the file associated with the SSD
ssd.file
"""
>>> /some/junk/test1.csv
"""

# show the data (or a sample in a Pandas dataframe)
ssd.df
"""
>>>
      SA1_MAINCODE_2011  SA1_7DIGITCODE_2011  SA2_MAINCODE_2011
0            10101100101              1100101          101011001
1            10101100102              1100102          101011001
2            10101100103              1100103          101011001
3            10101100104              1100104          101011001
4            10101100105              1100105          101011001
5            10101100106              1100106          101011001
...
"""

# we can assign DataNodes to the columns manually...
ssd.map(Column("name"), DataNode("name", "Person"))
"""
>>>
[
    Column(name, 1) -> DataNode(name, ClassNode(Person))
    Column(birth, 2) -> None
    Column(city, 3) -> None
    Column(state, 4) -> None
    Column(workplace, 5) -> None
]
"""

# we can also add transforms manually as lambda functions...
ssd.map(Column("birth"), DataNode("dob", "Person"), transform=pd.to_datetime)
"""
>>>
[
    Column(name, 1) -> DataNode(name, ClassNode(Person))
    Column(birth, 2) -> Transform(1) -> DataNode(dob, ClassNode(Person))
    Column(city, 3) -> None
    Column(state, 4) -> None
    Column(workplace, 5) -> None
]
"""

# we can view the transform table like so...
ssd.transform
"""
>>>
    {
        Transform(1): {
           'id': 1
           'source_columns': [1]
           'name': ''
           'label': ''
           'sql': 'select * from $1'
           'func': <function <lambda> at 0x76523401>
           'sample': [987345, 34598754, 345879, ...]
        }
    }
"""

# we can also try to auto-fill the SSD
ssd.predict()

# we can now see what the algorithm predicted
ssd
"""
>>>
[
    Column(name, 1) -> Transform(1) -> DataNode(name, ClassNode(Person))
    Column(birth, 2) -> DataNode(dob, ClassNode(Person))
    Column(city, 3) -> Transform(1) -> DataNode(name, ClassNode(City))
    Column(state, 4) -> Transform(4) -> DataNode(name, ClassNode(State))
    Column(workplace, 5) -> None
    ClassNode(Person) -> Link(is_a) -> ClassNode(OldPerson)
    ClassNode(Person) -> Link(has_a) -> ClassNode(Car)
]
"""

# we can correct them link by link. First by linking to a DataNode...
ssd.map(Column("workplace"), DataNode("name", "Organization"))
"""
>>> [
        Column(name, 1) -> Transform(1) -> DataNode(name, ClassNode(Person))
        Column(birth, 2) -> DataNode(dob, ClassNode(Person))
        Column(city, 3) -> Transform(fix_city, 1) -> DataNode(name, ClassNode(City))
        Column(state, 4) -> Transform(ident, 4) -> DataNode(name, ClassNode(State))
        Column(workplace, 5) -> DataNode(name, ClassNode(Organization))
        ClassNode(Person) -> Link(is_a) -> ClassNode(OldPerson)
        ClassNode(Person) -> Link(has_a) -> ClassNode(Car)
    ]
"""

# we can also add transforms
ssd.map(Column("city"), DataNode("name", "City"), transform=lambda c: str.lowercase(c))
"""
>>> [
        Column(name, 1) -> Transform(1) -> DataNode(name, ClassNode(Person))
        Column(birth, 2) -> DataNode(dob, ClassNode(Person))
        Column(city, 3) -> Transform(transform_1, 1) -> DataNode(name, ClassNode(City))
        Column(state, 4) -> Transform(ident, 4) -> DataNode(name, ClassNode(State))
        Column(workplace, 5) -> DataNode(name, ClassNode(Organization))
        ClassNode(Person) -> Link(is_a) -> ClassNode(OldPerson)
        ClassNode(Person) -> Link(has_a) -> ClassNode(Car)
    ]
"""

# we can chain these operators also...
(ssd.map(ClassNode("Person"), ClassNode("City"), relationship="lives-in")
    .map(ClassNode("Location"), ClassNode("City"), is_subclass=True)
    .map(ClassNode("Location"), ClassNode("State"), is_subclass=True))
"""
>>> [
        Column(name, 1) -> Transform(1) -> DataNode(name, ClassNode(Person))
        Column(birth, 2) -> DataNode(dob, ClassNode(Person))
        Column(city, 3) -> Transform(transform_1, 1) -> DataNode(name, ClassNode(City))
        Column(state, 4) -> Transform(ident, 4) -> DataNode(name, ClassNode(State))
        Column(workplace, 5) -> DataNode(name, ClassNode(Organization))
        ClassNode(Person) -> Link(is_a) -> ClassNode(OldPerson)
        ClassNode(Person) -> Link(has_a) -> ClassNode(Car)
        ClassNode(Person) -> Link(lives-in) -> ClassNode(City)
        ClassNode(Location) -> SubClass() -> ClassNode(City)
        ClassNode(Location) -> SubClass() -> ClassNode(State)
    ]
"""

# and also remove links...
ssd.remove(ClassNode("Person"), ClassNode("City"), relationship="lives-in")
"""
>>> [
        Column(name, 1) -> Transform(1) -> DataNode(name, ClassNode(Person))
        Column(birth, 2) -> DataNode(dob, ClassNode(Person))
        Column(city, 3) -> Transform(transform_1, 1) -> DataNode(name, ClassNode(City))
        Column(state, 4) -> Transform(ident, 4) -> DataNode(name, ClassNode(State))
        Column(workplace, 5) -> DataNode(name, ClassNode(Organization))
        ClassNode(Person) -> Link(is_a) -> ClassNode(OldPerson)
        ClassNode(Person) -> Link(has_a) -> ClassNode(Car)
        ClassNode(Location) -> SubClass() -> ClassNode(City)
        ClassNode(Location) -> SubClass() -> ClassNode(State)
    ]
"""

# you can launch the .dot viewer on linux, or perhaps a small graphviz web viewer
ssd.show()
"""
>>> Viewer launched at locahost:8080
"""

# You can then save to the ssd (json) file with the following.
# This will also run a consistency check...
ssd.save(file="some-file.ssd")
"""
>>> File check ok.
>>> File saved successfully to /something/something/some-file.ssd
"""

######
# SSD LOWER LEVEL
######

# we can also look at the mapping
ssd.mapping
"""
>>> {
        Transform(ident, 1): DataNode(name, Organization.name),
        Transform(ident, 6): DataNode(name, Person.name),
        Transform(fix_city, 2): DataNode(name, City.name),
        Transform(ident, 4): DataNode(name, State.name)
    }
"""

# we can also view the transforms in detail, as they would appear in the .ssd file.
ssd.transforms
"""
>>> {
    Transform(4): {
       'id': 4
       'source_columns': [1, 2]
       'name': 'something'
       'label': 's'
       'sql': 'select * from $1'
       'sample': [987345, 34598754, 345879, ...]
    },
    Transform(3): {
       'id': 3
       'source_columns': [1, 2]
       'name': 'something'
       'label': 's'
       'sql': 'select $1 from 4 * $2'
       'sample': [987345, 34598754, 345879, ...]
    },
    Transform(2): {
       'id': 1
       'source_columns': [1, 2]
       'name': 'something'
       'label': 's'
       'python': 'lambda x: str.lowercase(x)'
       'sample': [987345, 34598754, 345879, ...]
    }
}
"""

# the model object also has the SemanticModel as it would appear in the ssd file.
ssd.model
"""
>>> <SemanticModel: 34597898>
  {
    "nodes": {
       2: {
            "id": 2,
            "label": "Person",
            "type": "ClassNode"
       },
       3: {
            "id": 3,
            "label": "Car",
            "type": "ClassNode"
       },
       4: {
            "id": 4,
            "label": "Old Person",
            "type": "ClassNode"
       }
     },
    "links": [
       {
           "source": "Person"
           "target": "Old Person"
           "label": "is a"
           "type": "ObjectProperty"
       }, {
           "source": "Person"
           "target": "Old Person"
           "label": "is a"
           "type": "ObjectProperty"
       }, {
           "source": "Person"
           "target": "Old Person"
           "label": "is a"
           "type": "ObjectProperty"
       }
     ]
"""

# view the class nodes...
sm.model.nodes
"""
>>>
[ClassNode("Person", ["name", "address", "phone"]), ClassNode("Car", ["make", "model", "VIN"]),
 ClassNode("Old Person", ["name", "address", "phone"]),]
"""

# view the model links...
sm.model.links
"""
>> [Link("Old Person", "is_a", "Person"), Link("Person", "has_a", "Car")]
"""




#=========
# SSD COMPONENTS
#=========
# we can look at the columns individually...
ssd.columns
"""
>> [Column("name", 1), Column("birth", 2), Column("city", 3), Column("state", 4)]
"""

ssd.col("name")
"""
>>> <Column("name", 1)>
"""

ssd.col(1)
"""
>>> <Column("name", 1)>
"""

dir(ssd.columns[0])
"""
>>> ['__junk__', '___init__', 'name', 'file', 'sample', 'index']
"""

column = ssd.columns
column.name
"""
>>> "name"
"""
column.file
"""
>>> "/some/file/test.csv"
"""
