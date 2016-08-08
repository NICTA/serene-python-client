# external
import datetime
import logging
import os

# project
from dataint.wrappers.data_integration_py import SchemaMatcher

HERE = os.path.abspath(os.curdir)

# setting up the logging
log_file = 'app_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0')+'.log'
logging.basicConfig(filename=os.path.join(HERE, 'data', log_file),
                    level=logging.DEBUG, filemode='w',
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')



if __name__ == "__main__":
    dm = SchemaMatcher()
