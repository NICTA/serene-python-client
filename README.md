# Serene Python Client

The Serene Python Client project provides a Python client to execute various data integration tasks via defined APIs and to interpret responses.
File [example.py](https://github.com/NICTA/serene-python-client/blob/refactor/docs/example.py) contains some example uses.


### Prerequisites

To execute schema matching task, the server for [Serene](https://github.com/NICTA/serene) needs to be started.


### How to test
The nose unit tests module needs to be installed. To run the tests:
```
nosetests -v
```
### Run

To install the package 'serene-python-client', run
```
python setup.py install
```

Consult the [example.py](https://github.com/NICTA/serene-python-client/blob/refactor/doc/example.py) on how to use the library.

### Schema Matcher
To use the schema matcher directly, an example can be found here:
```
doc/matcher.py
```