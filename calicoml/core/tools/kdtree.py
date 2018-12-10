# Serialization-friendly wrapper for the Scikit KD-Tree

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from sklearn import neighbors


class KDTree(object):
    """Wrapper for sklearn.neighbors.KDTree"""
    def __init__(self, X, leaf_size=40, metric='minkowski'):
        self.tree = neighbors.KDTree(X, leaf_size, metric)
        self.X = X
        self.leaf_size = leaf_size
        self.metric = metric

    def kernel_density(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        return self.tree.kernel_density(*args, **kwargs)

    def query(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        return self.tree.query(*args, **kwargs)

    def query_radius(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        return self.tree.query_radius(*args, **kwargs)

    @staticmethod
    def prepare_kd_trees_dicts_for_serialization(kd_trees, serializer):
        """ shared utility to prepare dict with kd trees for serialization
        Please note that if tuple is used as key, back conversion of list to tuple is needed on
        deserialization """
        serialized_kd_trees_dict = {}
        for tree_key in kd_trees:
            serialized_kd_trees_dict[tree_key] = serializer.serialize(kd_trees[tree_key])
        result = list(serialized_kd_trees_dict.items())
        return result
