
from tensorflow.keras.metrics import MeanAbsoluteError as MAE
import tensorflow as tf

class PairMAE(tf.keras.metrics.Metric):
    
    metric = None
    # is possible to handle one of the two metrics, by changing the parameter metric
    def __init__(self, name, metric=0, **kwargs):
        super(PairMAE, self).__init__(name=name, **kwargs)
        
        self.metric = metric
        self.mae = MAE()
		

    def update_state(self, y_true, y_pred, sample_weight=None):
        # update the internal MeanAbsoluteError instance
        self.mae.update_state(y_true[:, self.metric], y_pred[:, self.metric])
        
    def result(self):
        return self.mae.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mae.reset_states()
