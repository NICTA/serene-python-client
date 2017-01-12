
import pandas as pd
import itertools as it
import os
import os.path
import numpy as np
import random


from collections import OrderedDict

def flatten(list2d):
    """Helper function to flatten a 2d list"""
    return [item for sublist in list2d for item in sublist]

class Column(object):
    """
        The Column object holds a single column from the dataset.
    """
    def __init__(self, filename, colname, title, lines):
        self.filename = filename   # filename to which the column was extracted from
        self.colname = colname     # column name
        self.title = title         # semantic label of the column (short title)
        self.lines = lines         # lines in the column

    def bagging(self, size=200, n=100):
        """
            Sample with replacement (generate n samples of [size] lines each)
        """
        X = [np.random.choice(self.lines, size) for x in range(n)]
        y = [self.title for _ in range(n)]
        return X, y

class Reader(object):
    """
    The Reader object extracts the columns from the museum dataset.

    A file in the museum dataset has an unusual structure where the
    file is one big row, with columns separated by new lines. The first
    line in the file indicates how many columns are present e.g.

    <num_cols>

    <col 1 URI label>
    <number of elements>
    <index> <value>
    ...

    <col 2 URI label>
    <number of elements>
    <index> <value>
    ...

    The index is some sort of lookup encoding for the value. Here we
    discard the index and just use the value, and the last path node
    of the URI as column semantic label.
    """
    def drop_index(self, line):
        """Drops the index number when reading the line"""
        d = it.dropwhile(lambda x: x != ' ', line)
        return ''.join(d).strip()

    def take_last(self, line, sep):
        """Takes the last segment from the uri as the label"""
        x = line.split(sep)
        return x[-1].strip()

    def read_file(self, filename):
        """Reads the columns out from the file structure, and returns them as a list of 'Column' objects

        Args: filename - the file in the eswc2015 format
        """
        columns = []
        with open(filename, 'r') as f:
            # read the number of columns in this file...
            n = int(f.readline())

            # read each column
            filename_short = self.take_last(filename, "/")
            for _ in range(n):
                f.readline() # dead space....
                lines = []
                indices = []
                # read the title
                raw_label = f.readline().strip()
                label = self.take_last(raw_label, "/")
                # read the number of elements in the column
                num = int(f.readline())
                # now read the rest of the lines...
                for _ in range(num):
                    s = f.readline()
                    index, value = s.split(" ", 1)
                    lines.append(value.strip())
                    indices.append(index)
                col = Column(filename_short, raw_label, label, lines)
                columns.append(col)

        return columns

    def read_dir(self, dir):
        """Reads all the files in directory 'dir' and
        extracts the columns into one large list...
        """
        all_cols = []
        for root, _, files in os.walk(dir):
            for f in files:
                # grab the extension...
                _, ext = os.path.splitext(f)
                # if it is a valid file, then extract all
                # the columns...
                if ext == '.txt':
                    cols = self.read_file(os.path.join(root, f))
                    all_cols += cols
        return files, all_cols

    def to_ml(self, all_cols, labels=None, size=20, n=100, train_frac=0.5, verbose=True):
        """
        Convert a list of columns ('Column' objects) into an X, y matrix
        
        """
        def diff(first, second):
            '''difference between two lists'''
            second = set(second)
            return [item for item in first if item not in second]
        
        def bag(cols):
            X = []
            y = []
            for col in cols:
                X_single, y_single = col.bagging(size, n)
                for x_s, y_s in zip(X_single, y_single):
                    flattened = [ord(char) for char in '\n'.join(x_s)]   # replace chars with their unicode indices
                    X.append(flattened)
                    y.append(label_lookup[y_s])
            return X, y

        if labels is None:
            labels = list(OrderedDict.fromkeys(x.title for x in all_cols))
            if verbose:
                print("semantic labels:")
                print(labels)
            if 'unknown' not in labels:
                if verbose: print('Adding \'unknown\' semantic label')
                labels = np.append(labels, 'unknown')
                if verbose:
                    print('Updated semantic labels:')
                    print(labels)
            
        label_lookup = {k: v for v, k in enumerate(labels)}
        if verbose:
            print("semantic label lookup table:")
            print(label_lookup)

        # here we randomly spit all_cols into training and test sets of columns 50/50, and then convert those sets of columns to X_train, y_train and X_test, y_test via bagging the rows from those columns
        train_cols = random.sample(all_cols, int(np.ceil(len(all_cols)*train_frac)))    #all_cols[::2]
        test_cols = diff(all_cols, train_cols)     #all_cols[1::2]
        X_train, y_train = bag(train_cols)
        X_test, y_test = bag(test_cols)
        if verbose:
            print("train data semantic labels:", sorted(set(y_train)))
            print("test data semantic labels: ", sorted(set(y_test)))

        return (X_train, y_train), (X_test, y_test), label_lookup, train_cols, test_cols

    def train_split(self, X, y, n=12000):
        """Returns (X_train, y_train), (X_test, y_test)"""
        return (X[n:], y[n:]), (X[:n], y[:n])


if __name__ == "__main__":

    # reader for the museum dataset
    reader = Reader()
    files, all_cols = reader.read_dir('data')
    print("Found", len(all_cols), "columns")

    # First we load up the data...
    print('Loading data...')
    (X_train, y_train), (X_test, y_test), label_lookup = reader.to_ml(
        all_cols, subsize, n_samples
    )
    print(label_lookup)
    # we can use the inverted lookup to look at the labels for analysis
    inverted_lookup = { v:k for k, v in label_lookup.items() }
    labels = label_lookup.keys()

    # next we prepare the data for the network...
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
#    print('samples of X_train rows:')
#    for i in np.random.randint(0,X_train.shape[0],size=5):
#        print([[chr(c) for c in X_train[i,:]], inverted_lookup[y_train[i]]])

    y_train = to_categorical(np.array(y_train), len(labels))
    y_test = to_categorical(np.array(y_test), len(labels))

    # now we build the model...