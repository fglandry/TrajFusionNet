import copy

from action_predict import ActionPredict
from utils.utils import *

    
class VanillaTransformer(ActionPredict):
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 num_hidden_units=128,
                 **kwargs):
        
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'
        self._mlp_units = num_hidden_units
        self._combined_model = True

    def get_data(self, data_type: str, data_raw: dict, 
                 model_opts: dict,
                 *args, **kwargs):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        #self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        combined_model_data = super(VanillaTransformer, self).get_data(
            data_type, data_raw, self.combined_model_ops, 
            combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
    
class BaseTransformerModel(ActionPredict):
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 num_hidden_units=128,
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d' # this is usually not used
        self._mlp_units = num_hidden_units
        self._combined_model = True

    def get_data(self, data_type: str, 
                 data_raw: dict, 
                 model_opts: dict,
                 submodels_paths: dict = None):
        """ Get processed data
        Args:
            data_type [str]: data split (train, val, test)
            data_raw [dict]: raw data dictionary
            model_opts [dict]: model options
            submodels_paths [dict]: dictionary containing paths to submodels saved on disk
        """

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        # self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d' # this is usually not used

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, 
                                               combined_model=self._combined_model,
                                               submodels_paths=submodels_paths)

        return combined_model_data

    def get_model(self, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    

class TrajFusionNet(BaseTransformerModel, 
                           ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformer(VanillaTransformer, 
                           ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformerb(VanillaTransformer, 
                              ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class SmallTrajFusionNet(BaseTransformerModel, 
                         ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class SmallTrajectoryTransformer(VanillaTransformer, 
                                 ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class SmallTrajectoryTransformerb(VanillaTransformer, 
                                 ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
