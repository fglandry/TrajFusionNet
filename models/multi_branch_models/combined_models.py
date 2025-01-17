import copy
import time
import yaml
import wget
from utils.utils import *
from models.base_models import AlexNet, C3DNet, CombinedModelV1Net, CombinedModelV2Net, \
                               CombinedModelV3Net, convert_to_fcn
from models.base_models import I3DNet, TransformerNet
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Average, Add
from tensorflow.keras.layers import ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, dot, concatenate, Activation
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from action_predict import ActionPredict
from utils.data_load import DataGenerator, get_generator

    
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

    def get_data(self, data_type, data_raw, model_opts,
                 *args, **kwargs):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        # assert self.combined_model_ops['obs_length'] == 16
        self.obs_length = self.combined_model_ops['obs_length']

        combined_model_data = super(VanillaTransformer, self).get_data(
            data_type, data_raw, self.combined_model_ops, 
            combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
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

    def get_data(self, data_type, data_raw, model_opts,
                 submodels_paths: dict = None):

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

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
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
