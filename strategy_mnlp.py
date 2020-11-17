from strategy_base import StrategyBase
import numpy as np
from libact.base.interfaces import ProbabilisticModel


def mnlp_score(probas, avg_type='mean'):
    if avg_type == 'mean':
        f_avg = np.mean
    elif avg_type == 'sum':
        f_avg = np.sum
    elif avg_type == 'max':
        f_avg = np.max
    
    return [f_avg(-np.log(e.max(axis=-1)), axis=-1) for e in probas]


class StrategyMNLP(StrategyBase):
    def __init__(self, avg_type='mean', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._avg_type = avg_type
    
    def calculate_score(self, X_pool):
        assert isinstance(self.model, ProbabilisticModel)
        
        probas = self.model.predict_proba(X_pool)
#         print('probas0', probas[0])
#         print('probas1', probas[1])
#         print('probas00', probas[0][0])
        
#         print('tprobas', type(probas))
#         print('tprobas0', type(probas[0]))
#         print('tprobas00', type(probas[0][0]))
        return mnlp_score(probas, self._avg_type)
