# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

from importlib import import_module

import six
import sklearn

import numpy as np
import pandas as pd
import sys
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# pylint: disable=no-name-in-module
from sklearn.tree._tree import Tree
from calicoml.core.tools.kdtree import KDTree


def get_class_by_name(name):
    """Gets a class object by its name, e.g. sklearn.linear_model.LogisticRegression"""
    if name.startswith('cid.analytics'):
        # We changed package names in March 2017. This preserves compatibility with old models.
        name = name.replace('cid.analytics', 'analytics.core')
    elif name.startswith('cid.'):
        name = name.replace('cid.', 'analytics.')

    module, class_name = name.rsplit('.', 1)
    return getattr(import_module(module), class_name)


def get_class_name(clazz):
    """Gets the fully qualified class name as a string"""
    return clazz.__module__ + '.' + clazz.__name__


def dump_serialized_object(obj):
    """Prints a more human-readable dump of a serialized (dictionary/array tree) object"""
    print(json.dumps(obj, indent=2, separators=(',', ': ')))


class Codec(object):
    """Provides functions for encoding and decoding objects"""
    def encode(self, obj, serializer):
        """\
        Encodes an object as Python primitives (lists, dictionaries, ints, strings, etc.)

        :param obj: object to serialize
        :param serializer: serializer to use if the codec needs to perform any serialization of its own
        (necessary, for example, when implementing recursive serialization of nested scikit estimators)
        :return: obj encoded as Python primitives

        """
        raise NotImplementedError()

    def decode(self, serialized_obj, serializer):
        """\
        Decodes an object from Python primitives into an object

        :param serialized_obj: object to deserialize
        :param serializer: serializer object to use for recursive deserialization
        :return: the deserialized object

        """
        raise NotImplementedError()


class TreeCodec(Codec):
    """\
    Codec for oddball classes that hide behind .so files and segfault when decoded with setattr
    """
    def encode(self, obj, serializer):
        result = {'__class__': get_class_name(obj.__class__)}
        result['__state__'] = self.tree_state_serialize(obj.__getstate__(), serializer)
        for attr in ['n_features', 'n_classes', 'n_outputs', 'max_depth']:
            result[attr] = serializer.serialize(getattr(obj, attr))
        return result

    def decode(self, serialized_obj, serializer):
        clazz = get_class_by_name(serialized_obj['__class__'])
        obj = clazz(*[serializer.deserialize(serialized_obj[k])
                      for k in ['n_features', 'n_classes', 'n_outputs']])

        state = self.tree_state_deserialize(serialized_obj['__state__'], serializer)
        state['max_depth'] = serialized_obj['max_depth']
        obj.__setstate__(state)
        return obj

    def tree_state_serialize(self, state, serializer):
        """
        :param state: Tree state to serialize
        :param serializer: Serializer being used
        :return:
        """
        result = {}
        result['node_count'] = state['node_count']
        result['values'] = serializer.serialize(state['values'])
        result['nodes'] = serializer.serialize([list(tup) for tup in state['nodes']])
        return result

    def tree_state_deserialize(self, state, serializer):
        # pylint: disable=no-member
        # pylint: disable=protected-access
        """Custom deserializer made for Trees
        {u'node_count': 0, u'values': {u'dtype': u'float64', u'elements': [], u'__class__': u'numpy.ndarray'}, u'nodes':
         {u'dtype': u'void448', u'elements': [], u'__class__': u'numpy.ndarray'}}
        """
        def force_node_types(node_val):
            """Converts void values to proper tuple"""
            return [convert(val) for val, convert in zip(node_val, [np.int64, np.int64, np.int64, np.float64,
                                                                    np.float64, np.int64, np.float64])]

        result = {}
        result['node_count'] = state['node_count']
        # pylint: disable=c-extension-no-member
        result['nodes'] = np.array([tuple(force_node_types(tup)) for tup in state['nodes']],
                                   dtype=sklearn.tree._tree.NODE_DTYPE)
        result['values'] = serializer.deserialize(state['values'])

        return result


class NamedAttributesCodec(Codec):
    """Simple codec which encodes/decodes objects based on getattr/setattr"""
    def __init__(self, attribute_names=None, constructor_argument_names=None, ascii_attributes=None):
        """\

        :param attribute_names: optional list of attributes names to encode/decode,
                default value "None" allows attributes to be dynamic (object dependent)
        :param constructor_argument_names: optional list of constructor parameters
        :param ascii_attributes: optional collection (that supports the "in" operator) of ascii attribute names
                                 as strings.
        """
        self.attribute_names = attribute_names
        self.constructor_argument_names = constructor_argument_names or []
        self.ascii_attributes = ascii_attributes or []

    def _potentially_decode_string(self, name, value):
        """\
        converts attribute value to ascii if the attribute should be ascii

        :param name: attribute/param name
        :param value: input value
        :return: Ascii-encoded value if it's an ascii attribute, otherwise input value
        """

        if name in self.ascii_attributes:
            if isinstance(value, (str, six.text_type)):
                return str(value)
            else:
                raise ValueError("NamedAttributesCodec._potentially_decode_string: bad value " + value +
                                 " of type " + str(type(value)) + ": expected unicode or ascii value")
        return value

    def encode(self, obj, serializer):
        result = {'__class__': get_class_name(obj.__class__)}

        if self.attribute_names is None:
            attribute_names_obj = []
            for k in obj.__dict__:
                attribute_names_obj.append(k)
            for attr in attribute_names_obj:
                result[attr] = serializer.serialize(getattr(obj, attr))
            return result

        for attr in self.attribute_names:
            result[attr] = serializer.serialize(getattr(obj, attr, None))
        return result

    def decode(self, serialized_obj, serializer):
        clazz = get_class_by_name(serialized_obj['__class__'])
        obj = clazz(**{k: self._potentially_decode_string(k, serializer.deserialize(serialized_obj[k]))
                       for k in self.constructor_argument_names})
        non_ctor_attributes = [name for name in self.attribute_names if name not in self.constructor_argument_names] \
            if self.attribute_names is not None else [k for k in serialized_obj
                                                      if k not in self.constructor_argument_names and k != '__class__']
        for attr in non_ctor_attributes:
            try:
                setattr(obj, attr, self._potentially_decode_string(attr, serializer.deserialize(serialized_obj[attr])))
            except AttributeError as e:
                temp = self._potentially_decode_string(attr, serializer.deserialize(serialized_obj[attr]))
                x = "NamedAttributesCodec.decode: Error setting attribute: " + attr + " to type " + str(type(temp)) +\
                    " and value:\n" + str(temp) + "\n" + "Exception message was: " + str(e)
                raise AttributeError(x)

        return obj


class IdentityCodec(Codec):
    """Identity codec: doesn't actually do any encoding/decoding. Only used for Python primitives"""
    def encode(self, obj, serializer):
        return obj

    def decode(self, serialized_obj, serializer):
        return serialized_obj


class BytesCodec(Codec):
    """Encodes Python3 bytes"""
    def encode(self, obj, serializer):
        # TODO: Converting bytes to an array of ints wastes memory
        return {'__class__': get_class_name(bytes),
                'bytes': [int(x) for x in obj]}

    def decode(self, serialized_obj, serializer):
        return bytes(serialized_obj['bytes'])


class NumpyPrimitiveCodec(Codec):
    """Identity codec for NumPy primitives"""
    def encode(self, obj, serializer):
        return obj.item()

    def decode(self, serialized_obj, serializer):
        return serialized_obj


class DictCodec(Codec):
    """Encodes/Decodes Python dictionaries"""
    def encode(self, obj, serializer):
        return {serializer.serialize(k): serializer.serialize(v)
                for k, v in obj.items()}

    def decode(self, serialized_obj, serializer):
        return {serializer.deserialize(k): serializer.deserialize(v)
                for k, v in serialized_obj.items()}


class TupleCodec(Codec):
    """Encodes/Decodes Python lists"""
    def encode(self, obj, serializer):
        return tuple([serializer.serialize(elt) for elt in obj])

    def decode(self, serialized_obj, serializer):
        return tuple([serializer.deserialize(elt) for elt in serialized_obj])


class ListCodec(Codec):
    """Encodes/Decodes Python lists"""
    def encode(self, obj, serializer):
        return [serializer.serialize(elt) for elt in obj]

    def decode(self, serialized_obj, serializer):
        return [serializer.deserialize(elt) for elt in serialized_obj]


class NumpyArrayCodec(Codec):
    """Encodes/Decodes Numpy arrays"""
    def encode(self, obj, serializer):
        return {'__class__': get_class_name(obj.__class__),
                'dtype': obj.dtype.name,
                'elements': [serializer.serialize(elt) for elt in obj]}

    def decode(self, serialized_obj, serializer):
        return np.array([serializer.deserialize(elt) for elt in serialized_obj['elements']],
                        dtype=serialized_obj['dtype'])


class DataFrameCodec(Codec):
    """Encodes/Decodes pandas DataFrames"""
    def encode(self, obj, serializer):
        # TODO FIXME: Saving the DataFrame as an embedded JSON string is gross, but that's what DataFrames provide
        # out of the box.
        # FIXME: pd.to_json() does not preserve order so we need to store that explicitly
        df = pd.DataFrame(obj)
        df['__order__'] = list(range(len(df)))
        return {'__class__': get_class_name(pd.DataFrame),
                'json_str': df.to_json()}

    def decode(self, serialized_obj, serializer):
        df = pd.read_json(serialized_obj['json_str'])
        df.sort_values(by=['__order__'], inplace=True)  # pylint: disable=no-member
        del df['__order__']
        return df


class KDTreeCodec(Codec):
    """Encodes/Decodes KDTrees"""
    def encode(self, obj, serializer):
        """Encodes KD tree for serializer"""
        return {'__class__': get_class_name(obj.__class__),
                'leaf_size': obj.leaf_size,
                'X': serializer.serialize(obj.X),
                'metric': obj.metric}

    def decode(self, serialized_obj, serializer):
        """Decodes KD tree for serializer"""
        return KDTree(serializer.deserialize(serialized_obj['X']),
                      leaf_size=serialized_obj['leaf_size'], metric=serialized_obj['metric'])


SVC_ATTRIBUTES = ('support_', 'support_vectors_', 'n_support_', '_dual_coef_', 'dual_coef_', '_intercept_',
                  '_sparse', 'shape_fit_', 'probA_', 'probB_', '_gamma', 'classes_',
                  'kernel', 'gamma', 'coef0', 'degree', 'probability', 'class_weight')
PCA_ATTRIBUTES = ('components_', 'explained_variance_ratio_', 'mean_',
                  'n_components_', 'noise_variance_',)
RFC_ATTRIBUTES = ('estimators_', 'classes_', 'n_classes_', 'oob_score', 'estimators_', 'n_outputs_', 'random_state',
                  'class_weight', 'estimator_params', 'base_estimator', 'bootstrap', 'n_features_')
DECISION_TREE_ATTRIBUTES = {'tree_', 'max_features_', 'classes_', 'n_classes_', 'n_features_', 'n_outputs_',
                            'random_state', 'class_weight', 'n_features_'}  # feature_importances done in Tree


class Serializer(object):
    """Serializes scikit estimators to/from Python primitives"""

    def __init__(self, debug_deserialize=False):
        self.codecs = {}
        self.debug_deserialize = debug_deserialize

        # Register codecs for primitives
        for primitive_class in [int, float, str, bool, type(None),
                                np.unicode_]:
            self.register_codec(primitive_class, IdentityCodec())

        if six.PY2:
            self.register_codec(six.text_type, IdentityCodec())  # unicode

        if six.PY3:
            self.register_codec(bytes, BytesCodec())

        # Register NumPy primitive codecs
        for numpy_primitive in [np.int32, np.int64, np.float64, np.float128]:
            self.register_codec(numpy_primitive, NumpyPrimitiveCodec())

        self.register_codec(np.ndarray, NumpyArrayCodec())
        self.register_codec(list, ListCodec())
        self.register_codec(tuple, TupleCodec())
        self.register_codec(dict, DictCodec())
        self.register_codec(SelectKBest, NamedAttributesCodec(['scores_', 'pvalues_', 'k']))
        self.register_codec(LogisticRegression, NamedAttributesCodec(['classes_', 'coef_', 'intercept_']))
        self.register_codec(GaussianNB, NamedAttributesCodec(['classes_', 'class_prior_', 'class_count_',
                                                              'theta_', 'sigma_']))
        self.register_codec(DictVectorizer, NamedAttributesCodec(['sparse', 'feature_names_',
                                                                  'vocabulary_']))
        self.register_codec(SVC, NamedAttributesCodec(SVC_ATTRIBUTES, ['kernel', 'probability'], ['kernel']))
        self.register_codec(PCA, NamedAttributesCodec(PCA_ATTRIBUTES))
        self.register_codec(RandomForestClassifier, NamedAttributesCodec(RFC_ATTRIBUTES, []))
        self.register_codec(DecisionTreeClassifier, NamedAttributesCodec(DECISION_TREE_ATTRIBUTES, [], []))
        self.register_codec(KDTree, KDTreeCodec())
        self.register_codec(Tree, TreeCodec())
        self.register_codec(Imputer, NamedAttributesCodec(['statistics_']))
        self.register_codec(Pipeline, NamedAttributesCodec(['steps'], constructor_argument_names=['steps']))
        self.register_codec(pd.DataFrame, DataFrameCodec())
        self.register_codec(StandardScaler, NamedAttributesCodec())

    def serialize(self, obj):
        """\
        Serializes an object into Python primitives using the available codecs.

        :param obj: object to serialize
        :return: the serialized object

        """
        if self.debug_deserialize:
            if not isinstance(self._get_codec(type(obj)), (IdentityCodec, NumpyPrimitiveCodec)):
                print("Serializing " + str(type(obj)) + " :")
                print(repr(obj))
                sys.stdout.flush()

        if hasattr(obj, 'serialize'):
            return obj.serialize(self)

        return self._get_codec(type(obj)).encode(obj, self)

    def _get_codec(self, clazz):
        if get_class_name(clazz) in self.codecs:
            return self.codecs[get_class_name(clazz)]
        else:
            raise ValueError('No codec available for class: {}. Available codecs: {}'.format(
                get_class_name(clazz), ', '.join(sorted(self.codecs.keys()))))

    def deserialize(self, serialized_obj):
        """\
        Deserializes an object from Python primitives using available codecs.

        :param obj: the serialized object
        :return: the deserialized object

        """
        if self.debug_deserialize:
            print("Deserializing:")
            dump_serialized_object(serialized_obj)
        if isinstance(serialized_obj, dict) and '__class__' in serialized_obj:
            clazz = get_class_by_name(serialized_obj['__class__'])
            if hasattr(clazz, 'deserialize'):
                # Override the codec and just use the provided serialization facilities
                return clazz.deserialize(serialized_obj, self)
            else:
                return self._get_codec(clazz).decode(serialized_obj, self)
        else:
            return self._get_codec(serialized_obj.__class__).decode(serialized_obj, self)

    def register_codec(self, clazz, codec):
        """\
        Registers a new codec with the serializer. The serializer will then be able to use the codec
        to encode/decode instances of a given class.

        :param clazz: class for which to register the codec
        :param codec: the codec to register
        :return: None

        """
        self.codecs[get_class_name(clazz)] = codec
        if get_class_name(clazz) not in self.codecs:
            raise ValueError("Serializer.register_codec Error: couldn't register codec for class " +
                             get_class_name(clazz))

    def dumps(self, obj, sort_keys=False):
        """ Returns dumped serialized object """
        return json.dumps(self.serialize(obj), sort_keys=sort_keys)
