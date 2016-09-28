
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
