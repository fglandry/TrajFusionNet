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


class CombinedModelV1(ActionPredict):
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 num_hidden_units=128,
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'
        self._mlp_units = num_hidden_units
        self._combined_model = True

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["c3d"])
        self.combined_model_ops.update(model_opts["base_transformer"])

        # Add model-specific parameters
        #assert len(self.c3d_model_ops['obs_input_type']) == 1
        assert self.combined_model_ops['obs_length'] == 16
        self.obs_length = self.combined_model_ops['obs_length']

        self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (112, 112)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super(CombinedModelV1, self).get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    # TODO: use keras function to load weights
    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        
        sequence_features_len = sum([d[-1] for d in data_params["data_sizes"]])
        
        model = CombinedModelV1Net(
            self._weights,
            self._freeze_conv_layers,
            self._dropout,
            self._dense_activation,
            sequence_features_len,
            data_params,
            model_opts,
            self._mlp_units)
        
        model.summary()
        return model

class CombinedModelV2(ActionPredict):
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 num_hidden_units=128,
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'
        self._mlp_units = num_hidden_units
        self._combined_model = True

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["c3d"])
        self.combined_model_ops.update(model_opts["base_transformer"])

        # Add model-specific parameters
        #assert len(self.c3d_model_ops['obs_input_type']) == 1
        assert self.combined_model_ops['obs_length'] == 16
        self.obs_length = self.combined_model_ops['obs_length']

        self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (112, 112)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super(CombinedModelV2, self).get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    # TODO: use keras function to load weights
    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        
        model = CombinedModelV2Net(
            data_params,
            model_opts,
            self._mlp_units)
        
        model.summary()
        return model
    
class CombinedModelV3(ActionPredict):
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 num_hidden_units=128,
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'
        self._mlp_units = num_hidden_units
        self._combined_model = True

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["vgg16"])
        self.combined_model_ops.update(model_opts["c3d"])
        self.combined_model_ops.update(model_opts["base_transformer"])

        # Add model-specific parameters
        #assert len(self.c3d_model_ops['obs_input_type']) == 1
        assert self.combined_model_ops['obs_length'] == 16
        self.obs_length = self.combined_model_ops['obs_length']

        self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (112, 112)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super(CombinedModelV3, self).get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
    
        model = CombinedModelV3Net(
            data_params,
            model_opts,
            self._mlp_units)
        
        model.summary()
        return model
    
class MultiBranchPytorchModelV1(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        #self.combined_model_ops.update(model_opts["vgg16"])
        #self.combined_model_ops.update(model_opts["c3d"])
        #self.combined_model_ops.update(model_opts["base_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        # assert self.combined_model_ops['obs_length'] == 16
        self.obs_length = self.combined_model_ops['obs_length']

        combined_model_data = super(VanillaTransformer, self).get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    

class MultiBranchPytorchModelV2(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["van"])
        self.combined_model_ops.update(model_opts["timesformer"])
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    

class MultiBranchPytorchModelV3(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["van"])
        self.combined_model_ops.update(model_opts["timesformer"])
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    

class MultiBranchPytorchModelV4(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["van"])
        self.combined_model_ops.update(model_opts["timesformer"])
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class MultiBranchPytorchModelV5(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["van"])
        self.combined_model_ops.update(model_opts["timesformer"])
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class MultiBranchPytorchModelV6(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["van"])
        self.combined_model_ops.update(model_opts["timesformer"])
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class MultiBranchPytorchModelV7(MultiBranchPytorchModelV6, 
                                ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class MultiSceneContextV1(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        #self.combined_model_ops.update(model_opts["van"])
        #self.combined_model_ops.update(model_opts["timesformer"])
        #self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class BaseTransformerCrossAttention(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class BaseTransformerCrossAttentionV2(ActionPredict):
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

    def get_data(self, data_type, data_raw, model_opts):

        # Get model opts specific to each sub-model
        self.combined_model_ops = copy.deepcopy(model_opts)
        self.combined_model_ops.update(model_opts["vanilla_transformer"])

        # Add model-specific parameters
        self.obs_length = self.combined_model_ops['obs_length']

        #self.combined_model_ops['normalize_boxes'] = False
        self.combined_model_ops['target_dim'] = (224, 224)
        self.combined_model_ops['process'] = False
        self.combined_model_ops['backbone'] = 'c3d'

        combined_model_data = super().get_data(data_type, data_raw, self.combined_model_ops, combined_model=self._combined_model)

        return combined_model_data

    def get_model(self, data_params, model_opts, data, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class BaseTransformerCrossAttentionV3(BaseTransformerCrossAttentionV2, 
                                      ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class BaseTransformerCrossAttentionV4(BaseTransformerCrossAttentionV2, 
                                      ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class BaseTransformerCrossAttentionV5(BaseTransformerCrossAttentionV2, 
                                      ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class BaseTransformerLSTMV1(BaseTransformerCrossAttentionV2, 
                                      ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class BaseTransformerLSTMV2(BaseTransformerCrossAttentionV2, 
                                      ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class BaseTransformerEncoderDecoderV1(BaseTransformerCrossAttentionV2, 
                                      ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajFusionNet(BaseTransformerCrossAttentionV2, 
                           ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class SmallTrajFusionNet(BaseTransformerCrossAttentionV2, 
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

class TrajectoryTransformerV2(VanillaTransformer, 
                              ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformerV2b(VanillaTransformer, 
                              ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformerV3(BaseTransformerCrossAttentionV2, 
                              ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformerV3b(BaseTransformerCrossAttentionV2, 
                               ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformerV3c(BaseTransformerCrossAttentionV2, 
                               ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class TrajectoryTransformerV3d(BaseTransformerCrossAttentionV2, 
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

class VanMultiscale(BaseTransformerCrossAttentionV2, 
                               ActionPredict):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
