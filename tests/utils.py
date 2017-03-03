"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Helper functions and classes for the test libraries
"""
import yaml
import os
import subprocess
import requests
import time
import tempfile


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

    def setup(self):
        stream = open(os.path.join('tests', 'config.yaml'), 'r')
        config = yaml.load(stream)
        launch_server = config['launch-test-server']

        # If launch-test-server is true, then this variable points to
        # the executable which will launch the server. If launch-test-server
        # is false then this is ignored.
        filename = config['launch-test-server-exe']

        # The location of the test server.
        host = config['test-server-host']
        port = config['test-server-port']
        version = "v1.0"

        # if we need to, we start the server with a temporary storage directory
        if launch_server:
            tmp_dir = "/tmp/test" #tempfile.mkdtemp()
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
