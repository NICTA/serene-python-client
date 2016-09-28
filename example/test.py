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

# ===========
#    SETUP
# ===========

import dataint

# start with a SemanticModeller object...
sm = dataint.SemanticModeller(
    config={
        "kmeans": 5,
        "threshold": 0.7,
        "search-depth": 100
    },
    ontologies=["/some/file/something.owl", "/some/other/file/something.owl"]
)
print(sm)


# build up a simple ontology
on = (dataint.Ontology()
      .prefix("asd", "asd:/9e8rgyusdgfiuhsdf.dfguhudfh")
      .uri("http://www.semanticweb.org/data_integration_project/report_example_ontology")
      .class_node("Person", ["name", "address", "phone"], prefix="xsd")
      .class_node(
        "Car",
        {
            "make": str,
            "model": str,
            "VIN": int
        })
      .relationship("Person", "owns", "Car")
      .class_node("Location")
      .class_node("City", parent="Location")
      .class_node("State", parent="Location")
      .relationship("City", "is_in", "State"))

print(on)

# add the ontology to the SemanticModeller
sm.add_ontology(on)

# list all the class nodes
print(sm.class_nodes)

