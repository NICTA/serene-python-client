"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Helper functions and classes for the test libraries
"""
import unittest2 as unittest
import yaml
import os
import subprocess
import requests
import time

from serene import Session, Serene


class SereneTestServer(object):
    """
    Builds a test server using the config file tests/config.yaml
    """
    def __init__(self):
        """
        Confirm to see that the files are there...
        """
        self.server = None
        self.host = None
        self.port = None
        self.SERENE_TEST_SERVER_PATH = 'SERENE_TEST_SERVER_PATH'

    def setup(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests")
        with open(os.path.join(path, 'config.yaml'), 'r') as stream:
            config = yaml.load(stream)

        # Test if we need to launch a server here...
        launch_server = config['launch-test-server']

        # If launch-test-server is true, then this variable points to
        # the executable which will launch the server. If launch-test-server
        # is false then this is ignored. If SERENE_TEST_SERVER_PATH is
        # defined this will override this variable
        if self.SERENE_TEST_SERVER_PATH in os.environ:
            filename = os.environ[self.SERENE_TEST_SERVER_PATH]
        else:
            filename = config['launch-test-server-exe']
        tmp_dir = config['launch-test-server-storage']

        # The location of the test server.
        host = config['test-server-host']
        port = config['test-server-port']
        version = "v1.0"

        # if we need to, we start the server with a temporary storage directory
        if launch_server:
            self.server = subprocess.Popen(
                [filename,
                 "--port", str(port),
                 "--host", host,
                 "--storage-path", tmp_dir],
                shell=False
            )

        uri = "http://{}:{}/{}".format(host, port, version)
        ok = self.check_connection(uri)
        if ok:
            self.port = port
            self.host = host
            print("Connected to {}".format(uri))

    @staticmethod
    def check_connection(uri):
        """
        Simple polling loop to test when the server is ready.

        :param uri:
        :return:
        """
        wait_time = 10
        for clock in range(wait_time):
            code = -1
            try:
                code = requests.get(uri).status_code
            except Exception:
                pass
            if code != 200:
                time.sleep(1)
                if clock == wait_time - 1:
                    msg = "Failed to connect to server at {}".format(uri)
                    raise Exception(msg)
                continue
            else:
                return True
        return False

    def tear_down(self):
        self.server.kill()


class TestWithServer(unittest.TestCase):
    """
    Tests the Endpoints class
    """
    _server = None
    _session = None
    _serene = None

    @classmethod
    def setUpClass(cls):
        cls._server = SereneTestServer()
        cls._server.setup()
        cls._serene = Serene(
            cls._server.host,
            cls._server.port,
            None,
            None,
            False
        )
        cls._session = cls._serene.session

    @classmethod
    def tearDownClass(cls):
        cls._server.tear_down()
