# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from calicoml.core.serialization.serializer import get_class_name

import pandas as pd


class DataSource(object):
    """\
    Generic data source that can be represented as a pandas DataFrame
    """
    def __init__(self, lazy=False):
        """\
        Constructor

        :param lazy: Boolean. If true (normal operation), the invoke the load function. If false (testing only),
        suppress the load function.
        """
        if not lazy:
            self.df = self.load()
            if self.df is None:
                raise ValueError('Could not load data: result is null')

    @property
    def info(self):
        """Returns implementation-specific metadata about this data source, as a dictionary"""
        return {}

    def load(self):
        """\
        Called when the dataframe is first needed.

        :return: loaded DataFrame
        """
        raise NotImplementedError()

    @property
    def dataframe(self):
        """Returns the DataFrame"""
        return self.df


class PandasDataSource(DataSource):
    """Wrapper around an existing pandas DataFrame"""
    def __init__(self, df, path=None):
        self.df = df
        self.path = path
        super(PandasDataSource, self).__init__()

    @property
    def info(self):
        return {'path': self.path}

    def load(self):
        """\
        Called when the dataframe is first needed.

        :return: loaded DataFrame
        """
        return self.df

    def serialize(self, serializer):
        """Serializes this PandasDataSource"""
        return {'__class__': get_class_name(PandasDataSource),
                'df': serializer.serialize(self.df),
                'path': serializer.serialize(self.path)}

    @staticmethod
    def deserialize(serialized_obj, serializer):
        """Deserializes a PandasDataSource"""
        return PandasDataSource(serializer.deserialize(serialized_obj['df']),
                                path=serializer.deserialize(serialized_obj['path']))


class SqlDataSource(DataSource):
    """DataSource from a SQL query"""
    def __init__(self, query, connection):
        """\

        :param query: SQL query
        :param connection: database connection/sqlalchemy engine to use
        """
        self.query = query
        self.connection = connection
        super(SqlDataSource, self).__init__()

    @property
    def info(self):
        return {'query': self.query}

    def load(self):
        """\
        Called when the dataframe is first needed.

        :return: loaded DataFrame
        """
        return pd.read_sql(self.query, self.connection)
