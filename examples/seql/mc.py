import pandas as pd

from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from .utils import SuffixTree, TransitionGraph


class MarkovChainModel(BaseEstimator):
    """Markov chain model with explicitley defined states.

    Parameters
    ----------
    base_esimtator : sklearn.base.ClassifierMixin
        Base classifier to construct emission probability matrix (i.e. probabilities of observing each of the states at each time). If not an instance of ``sklearn.calibration.CalibratedClassifierCV``, would be embedded into.

    target_label : str
        Label of targets column in the provided instances of ``pandas.DataFrame``.

    sequence_split_label : str
        Label of sequence splitter column in the provided instances of ``pandas.DataFrame``.

    max_state_length: int
        Maximum length of subsequence (explicit state in the model).

    min_support: int, or float, optional
        Minimum occurence of subsequence to take into account. This parameter can be:

            - None, in which case no filtering is done at all

            - An int, giving the exact number of minum subsequnce occurences in sequences

            - A float, giving the ratio of occurences from total occurrences of subsequences of length ``max_width`` in sequences
    """

    def __init__(self, base_estimator=None, target_label='target', sequence_split_label='sequence_id', max_state_length=10, min_support=None):
        self.target_label = target_label
        self.sequence_split_label = sequence_split_label
        if base_estimator is None:
            self.base_estimator = CalibratedClassifierCV(
                base_estimator=DecisionTreeClassifier(), method='isotonic')
        elif not isinstance(base_estimator, CalibratedClassifierCV):
            self.base_estimator = CalibratedClassifierCV(
                base_estimator=base_estimator)
        else:
            self.base_estimator = base_estimator

        self.suffix_tree = SuffixTree(
            max_width=max_state_length, min_support=min_support)

    def fit(self, data):
        """Fit the Markov model.

        Parameters
        ----------
        data : pandas.DataFrame
            Data sample to fit with.

        Returns
        -------
        self : MarkovChainModel
            Returns an instance of self.
        """

        sequences = [data[self.target_label][data[self.sequence_split_label] == val]
                     for val in data[self.sequence_split_label].unique()]

        self.suffix_tree.fit(sequences=sequences)
        self.transition_graph = TransitionGraph()
        self.transition_graph.fit(self.suffix_tree)

        # encoding the symbols to 0, 1, ...
        self.encoder = LabelEncoder()
        self.encoder.fit(data[self.target_label])
        self.symbol_encoder_mapping = {val: idx for idx,
                                       val in enumerate(self.encoder.classes_)}

        inputs, targets = self._process_sample(data)
        # fit the classifier (with calibrated output - probabilities)
        self.base_estimator.fit(inputs, targets)

        # calculate initial probability of each symbol
        self.init_probas = pd.Series(
            index=self.transition_graph.to_matrix().index).fillna(0)
        for symbol, proba in Counter([seq[0] for seq in sequences]).items():
            self.init_probas.ix[symbol] = proba

        # calculate symbol frequences in the sequences
        self.symbol_freqs = Counter(''.join(pd.concat(sequences).values))
        self.symbol_freqs = {
            key: val / sum(self.symbol_freqs.values()) for key, val in self.symbol_freqs.items()}

        return self

    def query(self, data, return_states_history=False):
        """Query new data sample with the fitted model.

        Parameters
        ----------
        data : pandas.DataFrame
            New sample to apply model to.

        return_states_history : bool, default False
            Specifies if the all the state history to be returned as well

        Returns
        -------
        last_state : char
            Last state forecasted with the model.
        state_history : list of str
            All the states predicted with the model.
        """
        inputs, targets = self._process_sample(data)
        probas = self.base_estimator.predict_proba(inputs)

        emission_matrix = _get_state_probabilities(
            probas, init_probas)

        states = self._viterbi(emission_matrix=emission_matrix)
        if return_states_history:
            return states[-1][-1], states[:-1]
        else:
            return states[-1][-1]

    @property
    def trainsition_matrix(self):
        return self.transition_graph.to_matrix()

    def _get_emission_matrix(self, probas, init_probas):
        state_probas = pd.DataFrame(columns=states, index=arange(len(probas)))
        for observation in range(len(probas)):
            for state in self.transition_graph.nodes():
                # the subsequnce is too short for the state
                if observation <= len(state) - 2:
                    continue
                # root node (beginning of a new sequence)
                if state == '^':
                    state_probas.loc[i, state] = 0.0
                else:
                    ps = []
                    # calculate probability of the subsequnce
                    # (assume all symbols are indepent)
                    symbol_probas_in_seq = [
                        self.symbol_freqs[symbol] for symbol in state]
                    # calculate probablity of observing the subsequence based
                    # on classifier
                    for history, p in enumerate(probas[(observation + max_length - 1 - len(state)):(observation + max_length - 1)]):
                        ps.append(p[symbol_probas_in_seq[history]])

                    state_probas.loc[observation, state] = np.prod(ps)

        return state_probas.fillna(0)  # / state_probas.sum(1)

    def _process_sample(self, data):
        targets = self.encoder.transform(data[self.target_label])
        inputs = data.drop(
            [self.target_label, self.sequence_split_label], axis=1)
        return inputs, targets

    def _viterbi(self, emission_matrix):
        sample_length = len(emission_matrix)
        trans1 = pd.DataFrame(index=emission_matrix.columns,
                              columns=range(sample_length))
        trans2 = pd.DataFrame(index=emission_matrix.columns,
                              columns=range(sample_length))
        trans1[0] = emission_matrix.ix[0] * self.init_probas
        trans2[0] = 0
        for i in range(1, sample_length):
            # TODO: check the order of multiplication
            t = trans1[i - 1].dot(self.transition_matrix)
            trans1[i] = emission_matrix.ix[i] * (t).max()
            trans2[i] = t.idxmax()
        zt = pd.Series(index=range(sample_length))
        xt = pd.Series(index=range(sample_length))
        zt[sample_length - 1] = trans1[sample_length - 1].idxmax()
        xt[sample_length - 1] = zt[sample_length - 1]
        for i in range(sample_length - 1, 1, -1):
            zt[i - 1] = trans2[i - 1][zt[i]]
            xt[i] = zt[i - 1]
        return xt
