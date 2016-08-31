import os
import pandas as pd

########################################################################################
# some formatting functions
def upperCase():
    return lambda x: x.upper()

def lowerCase():
    return lambda x: x.lower()

def formatNumbers(fmt='%.2f'):
    return lambda x: fmt % x

########################################################################################


class Column(object):
    """class column"""
    def __init__(self, orig_name, category, place, content):
        self.orig_name = orig_name
        self.category = category
        self.place = place
        self.content = content

    def __str__(self):
        try:
            cont = len(self.content)
        except:
            cont = 0
        return "Class Column(orig_name=%r, category=%r, place=%r, len(content)=%r)" %\
               (self.orig_name, self.category, self.place, cont)

class Dataset(object):
    def __init__(self):
        pass

class ColIndexGenerator(object):
    """Class to iterate over a list of columns and return chunks of consequent columns with the same category"""
    def __init__(self, lista, cat):
        self.cat = cat
        self.itera = lista # list of objects of class Column
        self.cur = 0
        self.length = len(lista)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ind = []
        while self.cur < self.length and self.itera[self.cur].category != self.cat: # skip columns which are not category
            self.cur += 1
        while self.cur < self.length and self.itera[self.cur].category == self.cat: # mark all columns which are this category
            ind.append(self.itera[self.cur].orig_name)
            self.cur += 1
        if len(ind) > 1: # we need only those columns which are consequent and have the same category
            return ind
        raise StopIteration

# define Transformation
#           name      : DateTime
#           columns   : a, b
#           condition : a.category == 'date' and b.category == 'time' and abs(a.place - b.place) == 1
#           expression: a.orig_name + ' ' + b.orig_name
def calculateDateTime(a, b):
    """calculate column DateTime given two columns a and b"""
    if a.category == 'date' and b.category == 'time' and abs(a.place - b.place) == 1:
        return a.orig_name + ' ' + b.orig_name


def concatenateColumns(new_name, cols, delimiter = ' '):
    """
    Create a function of a dataframework which calculates a new column in the data framework by concatenating
    the specified columns.

    Args:
        new_name: name for the new calculated column
        cols: list of columns which need to be concatenated
        delimiter: delimiter which is used when concatenating strings, it is of type string

    Returns:
        function(df)

    """
    def iter_fn(df):
        # concatenate all cols using delimiter
        df[new_name] = df[cols].apply(lambda x: delimiter.join(x), axis=1)
    str_cols = [str(col).replace(' ', '') for col in cols]
    iter_fn.__name__ = 'concatenateColumns_'+ '_'.join(str_cols)
    return iter_fn

def templateConcatenateConsequent(column_headers, category, new_name, delimiter = ' '):
    """
        Template function to concatenate columns from a specific category.
        Category, new_name and delimiter need to be specified by the user.

        Args:
            column_headers: list of objects of type Column
            category: learnt category to which template needs to be applied
            new_name: name to be assigned to the new calculated column
            delimiter: delimiter which will be used to concatenate columns

        Returns:
            list of functions which are to be applied to a Pandas dataframework

    """
    ind_gener = ColIndexGenerator(column_headers, category)
    return [concatenateColumns(new_name, cols, delimiter) for cols in ind_gener]


def formatColumn(format_func, col, new_name):
    """
        Args:
            format_func: function of x
            col: column name to which format_func needs to be applied
            new_name: name of the new calculated column

        Returns:
            function(df)

    """
    def iter_fn(df):
        df[new_name] = df[col].map(format_func)
    iter_fn.__name__ = 'formatColumn_'+format_func.__name__ + '_' + str(col).replace(' ', '')
    return iter_fn


def templateFormat(column_headers, category, format_func, new_name):
    """
        Template function to apply specified formatting to columns of a specific category

        Args:
            column_headers: list of objects of type Column
            category: learnt category to which template needs to be applied
            format_func: delimiter which will be used to concatenate columns
            new_name: name to be assigned to the new calculated column

        Returns:
            list of functions which are to be applied to a Pandas data framework

    """
    return [formatColumn(format_func, col.orig_name, new_name)\
            for col in column_headers if col.category == category]


def processTransformation(column_headers, transformation):
    """
        Apply transformation to the list of columns.

        Args:
            column_headers: list of objects of type Column
            transformation: a dictionary with fields
                                            category, template,
                                            delimiter, new_name,
                                            format_func
        Returns:
            list of functions which are to be applied to a Pandas data framework
    """
    # check transformation
    if type(transformation) != dict and transformation.keys() != set(['category', 'template',\
                                                                      'delimiter', 'new_name',\
                                                                      'format_func']):
        print("Error: transformation is not specified correctly.")
        return None

    if transformation['template'] == 'format':
        return templateFormat(column_headers, transformation['category'],
                              transformation['format_func'],transformation['new_name'])
    elif transformation['template'] == 'concatenate':
        # check if delimiter is properly specified
        if transformation['delimiter'] is None or type(transformation['delimiter']) != str:
            transformation['delimiter'] = ' ' # if delimiter is wrongly specified, default value is assigned
        return templateConcatenateConsequent(column_headers, transformation['category'],
                              transformation['new_name'], transformation['delimiter'])

    print("Error: template is not implemented. only templates 'format' and 'concatenate' are implemented.")
    return None

########################################################################################
if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'tests\dfat')
    to_process = []
    with open(os.path.join(path, 'Airports.csv')) as f:
        airports = pd.read_csv(f)
        airports_columns = []
        categories = {0: 'airport', 1: 'city', 2: 'country'}
        for i, col in enumerate(airports.columns.values):
            airports_columns.append(Column(col, categories[i], i, None))
        to_process.append(('airports',airports_columns))

    with open(os.path.join(path, 'Flights_Blue_Airline.csv')) as f:
        flights_blue = pd.read_csv(f)
        blue_columns = []
        categories = {0: 'flight', 1: 'name', 2: 'airport',
                      3: 'airport', 4: 'date', 5: 'time',
                      6: 'date', 7: 'time'}
        for i, col in enumerate(flights_blue.columns.values):
            blue_columns.append(Column(col, categories[i], i, None))
        to_process.append(('Flights_Blue_Airline',blue_columns))

    with open(os.path.join(path, 'Flights_Green_Airline.csv')) as f:
        flights_green = pd.read_csv(f)
        green_columns = []
        categories = {0: 'flight', 1: 'passport', 2: 'airport',
                      3: 'airport', 4: 'datetime', 5: 'datetime'}
        for i, col in enumerate(flights_green.columns.values):
            green_columns.append(Column(col, categories[i], i, None))
        to_process.append(('Flights_Green_Airline', green_columns))

    with open(os.path.join(path, 'personal_details.csv')) as f:
        person = pd.read_csv(f)
        person_columns = []
        categories = {0: 'name', 1: 'name', 2: 'phone',
                      3: 'address', 4: 'address', 5: 'address',
                      6: 'passport', 7: 'birthdate'}
        for i, col in enumerate(person.columns.values):
            person_columns.append(Column(col, categories[i], i, None))
        to_process.append(('personal_details',person_columns))

    with open(os.path.join(path, 'Ship_trips.csv')) as f:
        ship_trips = pd.read_csv(f)
        ships_columns = []
        categories = {0: 'ship', 1: 'port', 2: 'port',
                      3: 'date', 4: 'time',
                      5: 'date', 6: 'time'}
        for i, col in enumerate(ship_trips.columns.values):
            ships_columns.append(Column(col, categories[i], i, None))
        to_process.append(('Ship_trips', ships_columns))

    with open(os.path.join(path, 'Ship_trips_names.csv')) as f:
        ship_person = pd.read_csv(f)
        trips_columns = []
        categories = {0: 'ship', 1: 'name'}
        for i, col in enumerate(ship_person.columns.values):
            trips_columns.append(Column(col, categories[i], i, None))
        to_process.append(('Ship_trips_names', trips_columns))

    to_process = dict(to_process)

    # categories = []
    # for cols in to_process:
    #     categories += [col.category for col in cols]
    # categories = set(categories)
    categories = ['city', 'flight', 'name', 'country', 'address',\
                  'birthdate', 'datetime', 'phone', 'airport', \
                  'passport', 'time', 'date', 'ship', 'port']

################## specifying transformations
    # formatting transformations need to be done first
    format_func = lowerCase()
    format_func.__name__ = 'lower_case'
    transformation1 = {
        'category': 'name',
        'template': 'format',
        'delimiter': ' ',
        'new_name': None,
        'format_func': format_func
    }

    transformation2 = {
        'category': 'name',
        'template': 'concatenate',
        'delimiter': ' ',
        'new_name': 'full_name',
        'format_func': None
    }

    transformation3 = {
        'category': 'address',
        'template': 'concatenate',
        'delimiter': ' ',
        'new_name': 'full_name',
        'format_func': None
    }
#############################

    transformats = [transformation1, transformation2, transformation3]
    funcs = []

    for dat, column_headers in to_process.items():
        print("***Dataset %r" % dat)
        for transf in transformats:
            new_funcs = processTransformation(column_headers, transf)
            funcs += new_funcs
            for f in new_funcs:
                print(f)


