import warnings

import pandas as pd
import numpy as np
import scipy.spatial.distance as distance

from collections import Counter
from itertools import dropwhile
from anytree import Node, RenderTree, Resolver, ChildResolverError, LevelOrderIter
from networkx import DiGraph
from sklearn.base import BaseEstimator

from .exceptions import NotFittedError, FitFailedWarning


def _get_subsequence_counts(sequences, length, min_support=None):
    # type: (Union[List[str], List], Int,  Union[Int, Float]) -> OrderedDict
    """
    Counts all subsequnces of fixed length in the list of sequences.

    Parameters
    ----------
    sequences: list of pandas.Series
        List of sequences(possibly with diffferent lengths).

    length: int
        Length of subsequnces to take into account.

    min_support: int, or float, optional
        Minimum occurence of subsequence to take into account. This parameter can be:

            - None, in which case no filtering is done at all

            - An int, giving the exact number of minum subsequnce occurences in sequences

            - A float, giving the ratio of occurences from total occurrences of subsequences of length ``length`` in sequences

    Returns
    -------
    subsequences: collections.OrderedDict
        Dict with subsequnces as keys and corresponding counts as values.
    """
    subsequences = Counter([''.join(seq.values.tolist()[i:(i + length)])
                            for seq in sequences
                            for i in range(len(seq) - length + 1)
                            if len(seq) > 0])

    if min_support is not None:
        if min_support < 1.0 and min_support > 0.0:
            min_support = np.round(np.sum(subsequences.values()) * min_support)
        if min_support < 1 or np.round(min_support) != min_support:
            raise ValueError('Wrong value for min_support parameter!')
        for key, count in dropwhile(lambda key_count: key_count[1] >= min_support,
                                    subsequences.most_common()):
            del subsequences[key]

    return subsequences


class SuffixTree(Node, BaseEstimator):
    """Suffix tree representation of subsequences.

    Parameters
    ----------
    max_width: int
        Maximum width of the tree, or alternatively maximum length of subsequence.

    min_support: int, or float, optional
        Minimum occurence of subsequence to take into account. This parameter can be:

            - None, in which case no filtering is done at all

            - An int, giving the exact number of minum subsequnce occurences in sequences

            - A float, giving the ratio of occurences from total occurrences of subsequences of length ``max_width`` in sequences

    """

    def __init__(self, max_width, min_support=None):
        super().__init__(name='root')
        self._max_width = max_width
        self._min_support = min_support
        self._resolver = Resolver('name')
        self._fitted = False

    def _safe_resolve(self, sequence, parent=True):
        """Resolves path in the contructed tree.

        Paramters
        ---------
        sequence: list of chars
            Sequence defining the node.

        parent: bool, default True
            If return parent node or the node defined by the ``sequence`` parameter.

        Returns
        -------
        node: anytree.node
            Parent node or current node of the tree. None, if the node defined by the ``sequence`` parameter does not exist in the tree.
        """
        path = '/' + self.name + '/' + '/'.join([sequence[::-1][:i][::-1]
                                                 for i in range(1, len(sequence) + int(parent == False))])
        try:
            return self._resolver.get(self, path)
        except ChildResolverError as e:
            return None

    def _safe_resolve_parent(self, sequence):
        return self._safe_resolve(sequence, parent=True)

    def _safe_resolve_node(self, sequence):
        return self._safe_resolve(sequence, parent=False)

    def fit(self, sequences):
        """Fit the suffix tree.

        Paramters
        --------
        sequences: list of pandas.Series
            List of sequences(possibly with diffferent lengths).

        Returns
        -------
        self : SuffixTree
            Returns an instance of self.
        """
        all_sequences = np.hstack(sequences)
        self.count = len(all_sequences)
        self._symbols = np.unique(all_sequences)

        self.states = ['^']

        for width in range(1, self._max_width + 1):
            counts = _get_subsequence_counts(
                sequences, width, self._min_support)

            for sequence, count in counts.items():
                self.states.append(sequence)
                parent = self._safe_resolve_parent(sequence)
                thenode = Node(sequence, parent=parent, count=count)

        self._fitted = True
        return self

    def render(self):
        """Returns pretty representation of the tree."""
        if not self._fitted:
            raise NotFittedError()
        return RenderTree(self)


class TransitionGraph(DiGraph):
    """Graph representation for state transition probablities in sequences."""

    def __init__(self):
        super().__init__()
        self._fitted = False

    def fit(self, suffix_tree, sequences):
        """Fit the transition probabilities graph.

        Parameters
        ----------
        suffix_tree: SuffixTree
            A prefited suffix tree for the sequences.

        Returns
        -------
        self : TransitionGraph
            Returns an instance of self.
        """
        self._max_length = suffix_tree._max_width
        states = suffix_tree.states
        trans = pd.DataFrame(columns=states[1:], index=states)
        trans.fillna(0, inplace=True)
        for symbol, proba in Counter([seq.values[0] for seq in sequences]).items():
            if proba > 0:
                trans.ix['^'][symbol] = proba

        for state in states:
            state_len = len(state)
            for seq in sequences:
                seq = ''.join(seq)
                if state_len == self._max_length:
                    upto = len(seq)-state_len
                else:
                    upto = state_len
                for i in range(upto):
                    prev = seq[i:(i+state_len)]
                    for nxt_state_len in range(self._max_length, 0, -1):
                        # if nxt_state_len > state_len:
                        #     nxt = seq[(i+1-state_len):(i+nxt_state_len+1-state_len)]
                        # else:
                        #     nxt = seq[(i+1):(i+nxt_state_len+1)]
                        nxt = seq[(i+state_len+1-nxt_state_len):(i+state_len+1)]
                        if nxt in trans.columns:
                            break
                    trans.ix[prev][nxt] += 1

        # normalizing probabilities
        for state in states:
            trans.ix[state] /= trans.ix[state].sum()


        # storing to graph
        for state, transition in trans.iteritems():
            for next_state, proba in transition.iteritems():
                if proba > 0:
                    self.add_edge(state, next_state, {'proba': proba})

        self._fitted = True
        return self

    def calc_similarity(self, other):
        adj = self.to_matrix()
        cols = np.intersect1d(adj.columns, other.columns)
        rows = np.intersect1d(adj.index, other.index)
        return 1-distance.cosine(adj[cols].ix[rows].values.flatten(),
                                 other[cols].ix[rows].values.flatten())


    def to_matrix(self):
        """Returns adjacency matrix for the transition graph.

        The resulting matrix should sum up to 1 in each column.

        Returns
        -------
        adj_matrix: pandas.DataFrame
            Adjacency matrix corresponding to the learned transition probabilities grpah.
        """
        if not self._fitted:
            raise NotFittedError()
        return pd.DataFrame.from_dict(self.adj).\
            apply(lambda x: x.apply(
                lambda y: y['proba'] if pd.notnull(y) else 0))

    def get_stable_subgraph(self):
        """Extracts subgraph containing only maximum length states.

        Returns
        -------
        subgraph: networkx.DiGraph
            A subgraph with nodes representing only maximum length states.
        """
        if not self._fitted:
            raise NotFittedError()
        nbunch = []
        for node in self.nodes_iter():
            if len(node) == self._max_length:
                nbunch.append(node)
        return self.subgraph(nbunch)
