"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Holds Exceptions used for the Serene Python Client
"""


class SereneError(Exception):
    """Base class for exceptions in this project"""
    pass


class InternalError(SereneError):
    def __init__(self, expr, msg):
        """Exception raised by the Python client.

            Attributes:
                expr : input expression in which the error occurred
                msg  : explanation of the error
            """
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "InternalError in expression={} and message={}".format(self.expr, self.msg)

    def __str__(self):
        return self.__repr__()


class BadRequestError(SereneError):
    def __init__(self, expr, msg):
        """Exception raised for bad requests to APIs (400).

            Attributes:
                expr : input expression in which the error occurred
                msg  : explanation of the error
            """
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "BadRequestError in expression={} and message={}".format(self.expr, self.msg)

    def __str__(self):
        return self.__repr__()


class NotFoundError(SereneError):
    def __init__(self, expr, msg):
        """Exception raised if API returns not found error (404).

            Attributes:
                expr : input expression in which the error occurred
                msg  : explanation of the error
            """
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "NotFoundError in expression={} and message={}".format(self.expr, self.msg)

    def __str__(self):
        return self.__repr__()


class OtherError(SereneError):
    def __init__(self, status_code, expr, msg):
        """Exception raised if API returns some other error.

            Attributes:
                status_code : status code returned by API
                expr : input expression in which the error occurred
                msg  : explanation of the error
            """
        self.status_code = status_code
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "OtherError with status_code={} " \
               "in expression={} and message={}".format(self.status_code, self.expr, self.msg)

    def __str__(self):
        return self.__repr__()
