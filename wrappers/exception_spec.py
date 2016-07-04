class DataIntError(Exception):
    """Base class for exceptions in this project"""
    pass


class InternalDIError(DataIntError):
    def __init__(self, expr, msg):
        """Exception raised by the Python client.

            Attributes:
                expr -- input expression in which the error occurred
                msg  -- explanation of the error
            """
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "InternalDIError in expression=" + repr(self.expr) + " and message=" + repr(self.msg)

    def __str__(self):
        return self.__repr__()

class BadRequestError(DataIntError):
    def __init__(self, expr, msg):
        """Exception raised for bad requests to APIs.

            Attributes:
                expr -- input expression in which the error occurred
                msg  -- explanation of the error
            """
        # status_code = 400
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "BadRequestError in expression=" + repr(self.expr) + " and message=" + repr(self.msg)

    def __str__(self):
        return self.__repr__()


class NotFoundError(DataIntError):
    def __init__(self, expr, msg):
        """Exception raised if API returns not found error.

            Attributes:
                expr -- input expression in which the error occurred
                msg  -- explanation of the error
            """
        # status_code = 404
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "NotFoundError in expression=" + repr(self.expr) + " and message=" + repr(self.msg)

    def __str__(self):
        return self.__repr__()


class OtherError(DataIntError):
    def __init__(self, status_code, expr, msg):
        """Exception raised if API returns some other error.

            Attributes:
                status -- status code returned by API
                expr -- input expression in which the error occurred
                msg  -- explanation of the error
            """
        self.status_code = status_code
        self.expr = expr
        self.msg = msg

    def __repr__(self):
        return "OtherError with status_code=" + repr(self.status_code) + \
               " in expression=" + repr(self.expr) + " and message=" + repr(self.msg)

    def __str__(self):
        return self.__repr__()