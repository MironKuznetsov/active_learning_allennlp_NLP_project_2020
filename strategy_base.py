import random
from abc import abstractmethod

import numpy as np

from libact.base.dataset import ensure_sklearn_compat
from libact.base.interfaces import (ContinuousModel, ProbabilisticModel,
                                    QueryStrategy)


def subsample_corpus(X, amount):
    sample_indexes = set(random.sample(list(range(len(X))), amount))
    return np.array([e for i, e in enumerate(X) if i in sample_indexes]), list(sample_indexes)


def restore_predictions(preds, sample_indexes, real_len, fill_value=-10000):
    #results = np.full(real_len, fill_value)
    results = [fill_value] * real_len
    for num, i in enumerate(sample_indexes):
        results[i] = preds[num]

    return np.array(results).reshape(-1, 1)


class StrategyBase(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = kwargs.pop('model', None)

        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )

        self.model.train(self.dataset)
        
        self._subsample = kwargs.pop('subsample', 0)

    def update(self, new_indexes, labels):
        self.model.train(self.dataset, new_indexes)

    def make_query(self, return_score=False):
        unlabeled_entry_ids, X_pool = zip(*self.dataset.get_unlabeled_entries())
        X_pool = ensure_sklearn_compat(X_pool)

        if self._subsample:
            subsample = (int(X_pool.shape[0] * self._subsample)
                         if isinstance(self._subsample, float)
                         else self._subsample)
            X_pool_subsampled, sample_indexes = subsample_corpus(X_pool, subsample)

            score = self.calculate_score(X_pool_subsampled)
            score = restore_predictions(score, sample_indexes, X_pool.shape[0])
        else:
            score = self.calculate_score(X_pool)

        # print(X_pool)
        print('score', max(score))
        print('score type', type(score))
        ask_id = np.argmax(score)

        if return_score:
            return unlabeled_entry_ids[ask_id], \
                   list(zip(unlabeled_entry_ids, score))
        else:
            return unlabeled_entry_ids[ask_id]

    @abstractmethod
    def calculate_score(self, X_pool):
        pass
