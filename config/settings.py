# configuration settings for the schema matcher server
schema_matcher_server = {
    'uri': "http://localhost:8080/v1.0/"    # uri for the API
    , 'auth': None                          # default authentication tuple ('user', 'pass')
    , 'cert': None                          # SSL certificate default
    , 'trust_env': False                    # trust environment settings for proxy configuration, default authentication, ...

}

# configuration settings for the data quality scoring server
data_quality_server = {
    'uri': ""
}