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
from utils.data_load import DataGenerator, get_generator, get_static_context_data


class BaseTransformer(ActionPredict):
    """
    """
    def __init__(self,
                 num_hidden_units=128,
                 **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: 
        """
        super().__init__(**kwargs)
        # Network parameters
        self._mlp_units = num_hidden_units

    def get_model(self, data_params, *args, **kwargs):
        model_opts = kwargs["model_opts"]
        network_inputs = []
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        core_size = len(data_sizes)

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i],
                                        name='input_' + data_types[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(network_inputs)
        else:
            inputs = network_inputs[0]

        net_model = TransformerNet.build_model(
            model_opts,
            inputs,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[self._mlp_units],
            mlp_dropout=0.4,
            dropout=0.25,
        )
        return net_model


class SingleRNN(ActionPredict):
    """ A simple recurrent network """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn_type = cell_type

    def get_model(self, data_params, *args, **kwargs):
        network_inputs = []
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        core_size = len(data_sizes)
        _rnn = self._gru if self._rnn_type == 'gru' else self._lstm

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i],
                                        name='input_' + data_types[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(network_inputs)
        else:
            inputs = network_inputs[0]

        encoder_output = _rnn(name='encoder')(inputs)

        encoder_output = Dense(1, activation='sigmoid',
                               name='output_dense')(encoder_output)
        net_model = Model(inputs=network_inputs,
                          outputs=encoder_output)

        return net_model


class StackedRNN(ActionPredict):
    """ A stacked recurrent prediction model based on
    Yue-Hei et al. "Beyond short snippets: Deep networks for video classification."
    CVPR, 2015." """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params, *args, **kwargs):
        network_inputs = []
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        core_size = len(data_sizes)
        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(network_inputs)
        else:
            inputs = network_inputs[0]

        encoder_output = self.create_stack_rnn(core_size)(inputs)
        encoder_output = Dense(1, activation='sigmoid',
                               name='output_dense')(encoder_output)
        net_model = Model(inputs=network_inputs,
                          outputs=encoder_output)
        return net_model


class MultiRNN(ActionPredict):
    """
    A multi-stream recurrent prediction model inspired by
    Bhattacharyya et al. "Long-term on-board prediction of people in traffic
    scenes under uncertainty." CVPR, 2018.
    """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params, *args, **kwargs):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i])(network_inputs[i]))

        if len(encoder_outputs) > 1:
            encodings = Concatenate(axis=1)(encoder_outputs)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense')(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        return net_model


class HierarchicalRNN(ActionPredict):
    """
    A Hierarchical recurrent prediction model inspired by
    Du et al. "Hierarchical recurrent neural network for skeleton
    based action recognition." CVPR, 2015.
    """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params, *args, **kwargs):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        for i in range(core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(
                self._rnn(name='enc_' + data_types[i], r_sequence=True)(network_inputs[i]))

        if len(network_inputs) > 1:
            inputs = Concatenate(axis=2)(encoder_outputs)
        else:
            inputs = network_inputs[0]

        second_layer = self._rnn(name='final_enc', r_sequence=False)(inputs)

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense')(second_layer)
        net_model = Model(inputs=network_inputs,
                          outputs=model_output)

        return net_model


class SFRNN(ActionPredict):
    """
    Pedestrian crossing prediction based on
    Rasouli et al. "Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs."
    BMVC, 2020. The original code can be found at https://github.com/aras62/SF-GRU
    """
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru', **kwargs):
        """
        Class init function
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell

    def get_model(self, data_params, *args, **kwargs):
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        return_sequence = True
        num_layers = len(data_sizes)

        for i in range(num_layers):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))

            if i == num_layers - 1:
                return_sequence = False

            if i == 0:
                x = self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i])
            else:
                x = Concatenate(axis=2)([x, network_inputs[i]])
                x = self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(x)

        model_output = Dense(1, activation='sigmoid', name='output_dense')(x)

        net_model = Model(inputs=network_inputs, outputs=model_output)

        return net_model


class C3D(ActionPredict):
    """
    C3D code based on
    Tran et al. "Learning spatiotemporal features with 3d convolutional networks.",
    CVPR, 2015. The code is based on implementation availble at
    https://github.com/adamcasson/c3d
    """
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
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

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16
        self.obs_length = model_opts['obs_length']

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (112, 112)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super(C3D, self).get_data(data_type, data_raw, model_opts)

    # TODO: use keras function to load weights
    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            weights_url = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'
            wget.download(weights_url, self._weights)
        net_model = C3DNet(freeze_conv_layers=self._freeze_conv_layers,
                           dropout=self._dropout,
                           dense_activation=self._dense_activation,
                           include_top=True,
                           weights=self._weights)
        net_model.summary()

        return net_model


class I3D(ActionPredict):
    """
    A single I3D method based on
    Carreira et al. "Quo vadis, action recognition? a new model and the kinetics dataset."
    CVPR 2017. This model is based on the original code published by the authors which
    can be found at https://github.com/deepmind/kinetics-i3d
    """
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
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
        self._backbone = 'i3d'

    def get_data(self, data_type, data_raw, model_opts):
        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        model_opts['process'] = False
        model_opts['backbone'] = 'i3d'
        return super(I3D, self).get_data(data_type, data_raw, model_opts)

    # TODO: use keras function to load weights
    def get_model(self, data_params, *args, **kwargs):
        # TODO: use keras function to load weights

        if 'flow' in data_params['data_types'][0]:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            self._weights='weights/i3d_flow_weights.h5'
            num_channels = 2
        else:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            num_channels = 3
            self._weights='weights/i3d_rgb_weights.h5'
            
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            wget.download(weights_url, self._weights)
        
        net_model = I3DNet(freeze_conv_layers=self._freeze_conv_layers, weights=self._weights,
                           dense_activation=self._dense_activation, dropout=self._dropout,
                           num_channels=num_channels, include_top=True)

        net_model.summary()
        return net_model


class TwoStreamI3D(ActionPredict):
    """
    Two-stream 3D method based on
    Carreira et al. "Quo vadis, action recognition? a new model and the kinetics dataset."
    CVPR 2017. This model is based on the original code published by the authors which
    can be found at https://github.com/deepmind/kinetics-i3d
    """
    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights_rgb='weights/i3d_rgb_weights.h5',
                 weights_flow='weights/i3d_flow_weights.h5',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights_rgb: Pre-trained weights for rgb stream.
            weights_flow: Pre-trained weights for optical flow stream.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights_rgb = weights_rgb
        self._weights_flow = weights_flow
        self._weights = None
        self._backbone = 'i3d'

    def get_data(self, data_type, data_raw, model_opts):
        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        model_opts['process'] = False
        model_opts['backbone'] = 'i3d'
        return super(TwoStreamI3D, self).get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        # TODO: use keras function to load weights
        if 'flow' in data_params['data_types'][0]:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            num_channels = 2
        else:
            weights_url = 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
            num_channels = 3
        
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            wget.download(weights_url, self._weights)

        net_model = I3DNet(freeze_conv_layers=self._freeze_conv_layers, weights=self._weights,
                           dense_activation=self._dense_activation, dropout=self._dropout,
                           num_channels=num_channels, include_top=True)

        net_model.summary()
        return net_model

    def train(self, data_train,
              data_val=None,
              batch_size=4,
              epochs=60,
              lr=0.000005,
              optimizer='sgd',
              learning_scheduler=None,
              model_opts=None):

        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}

        model_opts['reshape'] = True
        local_params = {k: v for k, v in locals().items() if k != 'self'}

        #####  Optical flow model
        self.train_model('opt_flow', **local_params)

        ##### rgb model
        self.train_model('rgb', **local_params)

        # Save settings
        model_opts_path, saved_files_path = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        return saved_files_path

    def train_model(self, model_type, path_params, learning_scheduler, data_train, data_val,
                    optimizer, batch_size, model_opts,  epochs, lr, **kwargs):
        """
        Trains a single model
        Args:
            model_type: The type of model, 'rgb' or 'opt_flow'
            path_params: Parameters for generating paths for saving models and configurations
            callbacks: List of training call back functions
            model_type: The model type, 'rgb' or 'opt_flow'
            For other parameters refer to train()
        """
        learning_scheduler = learning_scheduler or {}
        self._weights = self._weights_rgb if model_type == 'rgb' else self._weights_flow

        _opts = model_opts.copy()
        if model_type == 'opt_flow':
            _opts['obs_input_type'] = [_opts['obs_input_type'][0] + '_flow']

        # Read train data
        data_train = self.get_data('train', data_train, {**_opts, 'batch_size': batch_size})
        if data_val is not None:
            data_val = self.get_data('val', data_val, {**_opts, 'batch_size': batch_size})
            data_val = data_val['data']
            if self._generator:
                data_val = data_val[0]

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model = self.get_model(data_train['data_params'])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        data_path_params = {**path_params, 'sub_folder': model_type}
        model_path, _ = get_path(**data_path_params, file_name='model.h5')
        callbacks = self.get_callbacks(learning_scheduler,model_path)

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('{} train model is saved to {}'.format(model_type, model_path))
            train_model.save(model_path)
        # Save training history
        history_path, saved_files_path = get_path(**data_path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)


    def test(self, data_test, model_path=''):
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        # Evaluate rgb model
        test_data_rgb = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
        rgb_model = load_model(os.path.join(model_path, 'rgb', 'model.h5'))
        results_rgb = rgb_model.predict(test_data_rgb['data'][0], verbose=1)

        model_opts['obs_input_type'] = [model_opts['obs_input_type'][0] + '_flow']
        test_data_flow = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
        opt_flow_model = load_model(os.path.join(model_path, 'opt_flow', 'model.h5'))
        results_opt_flow = opt_flow_model.predict(test_data_flow['data'][0], verbose=1)

        # Average the predictions for both streams
        results = (results_rgb + results_opt_flow) / 2.0
        gt = test_data_rgb['data'][1]

        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                          recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        # with open(save_results_path, 'w') as fid:
        #     yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

# weights="weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
class Static(ActionPredict):
    """
    A static model which uses features from the last convolution
    layer and a dense layer to classify
    """
    def __init__(self,
                 dropout=0.0,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights="weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", # used to be imagenet
                 num_classes=1,
                 backbone='vgg16',
                 global_pooling='avg',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
            num_classes: Number of activity classes to predict.
            backbone: Backbone network. Only vgg16 is supported.
            global_pooling: Global pooling method used for generating convolutional features
        """
        super().__init__(**kwargs)
        # Network parameters

        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._num_classes = num_classes
        self._pooling = global_pooling
        self._conv_models = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'alexnet': AlexNet}
        self._backbone = backbone

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates train/test data
        :param data_raw: The sequences received from the dataset interface
        :param model_opts: Model options:
                            'obs_input_type': The types of features to be used for train/test. The order
                                            in which features are named in the list defines at what level
                                            in the network the features are processed. e.g. ['local_context',
                                            pose] would behave different to ['pose', 'local_context']
                            'enlarge_ratio': The ratio (with respect to bounding boxes) that is used for processing
                                           context surrounding pedestrians.
                            'pred_target_type': Learning target objective. Currently only supports 'crossing'
                            'obs_length': Observation length prior to reasoning
                            'time_to_event': Number of frames until the event occurs
                            'dataset': Name of the dataset

        :return: Train/Test data
        """
        # Stack of 5-10 optical flow. For each  image average results over two
        # branches and average for all samples from the sequence
        # single images and stacks of optical flow

        # assert len(model_opts['obs_input_type']) == 1
        self._generator = model_opts.get('generator', True)
        model_opts['normalize_boxes'] = False if not model_opts['model'] in ["VAN", "SmallVAN"] else True
        process = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        feature_type = model_opts['obs_input_type'][0]

        assert feature_type in ['local_box', 'local_context', 'scene', 'scene_context', 
                                'scene_context_with_flow_optical', 
                                'scene_context_with_flow_optical_v2',
                                'scene_context_with_flow_optical_v3',
                                'scene_context_with_flow_optical_v4',
                                'scene_context_with_flow_optical_v5',
                                'scene_context_with_flow_optical_v6',
                                'scene_context_with_segmentation',
                                'scene_context_with_segmentation_v0',
                                'scene_context_with_segmentation_v2',
                                'scene_context_with_segmentation_v3',
                                'scene_context_with_segmentation_v4',
                                'scene_context_with_running_ped',
                                'scene_context_with_running_ped_doubled',
                                'scene_context_with_ped',
                                'scene_context_with_running_ped_v2',
                                'scene_context_with_running_ped_v2_doubled'
                                ]

        _data_samples = {}
        _data_samples['crossing'] = data['crossing']
        data_type_sizes_dict = {}


        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join(['local_context', aux_name, str(eratio)]) if feature_type == 'local_context' \
                           else '_'.join([feature_type, aux_name])
        path_to_features, _ = get_path(save_folder=save_folder_name,
                                          dataset=dataset,
                                          save_root_folder='data/features')
        data_gen_params = {'data_type': data_type, 'save_path': path_to_features,
                           'crop_type': 'none', 'process': process}

        if feature_type == 'local_box':
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif feature_type == 'local_context':
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'with_running_ped' in feature_type: # added
            data_gen_params['crop_type'] = 'remove_running_ped'
        elif 'with_ped' in feature_type: # added:
            data_gen_params['crop_type'] = 'keep_ped'
        elif 'scene_context' in feature_type and 'segmentation' not in feature_type: # added
            data_gen_params['crop_type'] = 'remove_ped'

        _data_samples[feature_type], feat_shape = get_static_context_data(
            self, model_opts, data, data_gen_params, feature_type
        )
        """
        # Get some settings from config
        concatenate_frames = "concatenate_frames" in model_opts and model_opts["concatenate_frames"]["enabled"]
        add_optical_flow = "flow_optical" in feature_type
        data_gen_params['concatenate_frames'] = concatenate_frames
        data_gen_params['feature_type'] = feature_type
        data_gen_params['is_feature_static'] = True

        # Keep latest element in sequence (model will be run on one frame)
        if not concatenate_frames and not add_optical_flow:
            for k, v in data.items():
                if 'act' not in k:
                    if len(v.shape) == 3:
                        data[k] = np.expand_dims(v[:, -1, :], axis=1)
                    else:
                        data[k] = np.expand_dims(v[:, -1], axis=-1)  
        
        _data_samples[feature_type], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                      data['box_org'],
                                                                      data['ped_id'],
                                                                      **data_gen_params)
        """
    
        if not self._generator:
            _data_samples[feature_type] = np.squeeze(_data_samples[feature_type])
        data_type_sizes_dict[feature_type] = feat_shape[1:]

        # create the final data file to be returned
        if self._generator:
            """
            _data_rgb = (DataGenerator(data=[_data_samples[feature_type]],
                                   labels=data['crossing'],
                                   data_sizes=[data_type_sizes_dict[feature_type]],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), _data_samples['crossing'])  # set y to None
            """
            _data_rgb = get_generator(
                            _data=[_data_samples[feature_type]],
                            data=data,
                            data_sizes=[data_type_sizes_dict[feature_type]],
                            process=process,
                            global_pooling=self._global_pooling,
                            model_opts=model_opts,
                            data_type=data_type,
                        )
            
        else:
            _data_rgb = (_data_samples[feature_type], _data_samples['crossing'])

        return {'data': _data_rgb,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': [feature_type],
                                'data_sizes': [data_type_sizes_dict[feature_type]]},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params, *args, **kwargs):
        data_size = data_params['data_sizes'][0]
        context_net = self._conv_models[self._backbone](input_shape=data_size,
                                                        include_top=False, weights=self._weights,
                                                        pooling=self._pooling)
        output = Dense(self._num_classes,
                       activation=self._dense_activation,
                       name='output_dense')(context_net.outputs[0])
        net_model = Model(inputs=context_net.inputs[0], outputs=output)

        net_model.summary()
        return net_model


class ConvLSTM(ActionPredict):
    """
    A Convolutional LSTM model for sequence learning
    """
    def __init__(self,
                 global_pooling='avg',
                 filter=64,
                 kernel_size=1,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Global pooling method used for generating convolutional features
            filter: Number of conv filters
            kernel_size: Kernel size of conv filters
            dropout: Dropout value for fc6-7 only for alexnet.
            recurrent_dropout: Recurrent dropout value
        """
        super().__init__(**kwargs)
        # Network parameters
        self._pooling = global_pooling
        self._filter = filter
        self._kernel_size = kernel_size
        self._dropout = dropout
        self._recurrent_dropout = recurrent_dropout
        self._backbone = ''

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        model_opts['normalize_boxes'] = False
        model_opts['process'] = False
        model_opts['backbone'] = ''
        return super(ConvLSTM, self).get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        data_size = data_params['data_sizes'][0]
        data_type = data_params['data_types'][0]

        x_in = Input(shape=data_size, name='input_' + data_type)
        convlstm = ConvLSTM2D(filters=self._filter, kernel_size=self._kernel_size,
                              kernel_regularizer=self._regularizer, recurrent_regularizer=self._regularizer,
                              bias_regularizer=self._regularizer, dropout=self._dropout,
                              recurrent_dropout=self._recurrent_dropout)(x_in)
        if self._pooling == 'avg':
            out = GlobalAveragePooling2D()(convlstm)
        elif self._pooling == 'max':
            out = GlobalMaxPooling2D()(convlstm)
        else:
            out = Flatten(name='flatten')(convlstm)

        _output = Dense(1, activation='sigmoid', name='output_dense')(out)
        net_model = Model(inputs=x_in, outputs=_output)
        net_model.summary()
        return net_model


class ATGC(ActionPredict):
    """
    This is an implementation of pedestrian crossing prediction model based on
    Rasouli et al. "Are they going to cross? A benchmark dataset and baseline
    for pedestrian crosswalk behavior.", ICCVW, 2017.
    """
    def __init__(self,
                 dropout=0.0,
                 freeze_conv_layers=False,
                 weights='imagenet',
                 backbone='alexnet',
                 global_pooling='avg',
                 **kwargs):
        """
            Class init function
            Args:
                dropout: Dropout value for fc6-7 only for alexnet.
                freeze_conv_layers: If set true, only fc layers of the networks are trained
                weights: Pre-trained weights for networks.
                backbone: Backbone network. Only vgg16 is supported.
                global_pooling: Global pooling method used for generating convolutional features
        """
        super().__init__(**kwargs)
        self._dropout = dropout
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._pooling = global_pooling
        self._conv_models = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50, 'alexnet': AlexNet}
        self._backbone = backbone

    def get_scene_tags(self, traffic_labels):
        """
        Generates a 1-hot vector for traffic labels
        Args:
            traffic_labels: A dictionary of traffic labels read from the label
            Original labels are:
            'ped_crossing','ped_sign','traffic_light','stop_sign','road_type','num_lanes'
             traffic_light: 0: n/a, 1: red, 2: green
             street: 0, parking_lot: 1, garage: 2
             final_output: [narrow_road, wide_road, ped_sign, ped_crossing,
                            stop_sign, traffic_light, parking_lot]
        Returns:
            List of 1-hot vectors for traffic labels
        """

        scene_tags = []
        for seq in traffic_labels:
            step_tags = []
            for step in seq:
                tags = [int(step[0]['num_lanes'] <= 2), int(step[0]['num_lanes'] > 2),
                        step[0]['ped_sign'], step[0]['ped_crossing'], step[0]['stop_sign'],
                        int(step[0]['traffic_light'] > 0), int(step[0]['road_type'] == 1)]
                step_tags.append(tags)
            scene_tags.append(step_tags)
        return scene_tags

    def get_data_sequence(self, data_type, data_raw, opts):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')

        d = {'box': data_raw['bbox'].copy(),
             'crossing': data_raw['activities'].copy(),
             'walking': data_raw['actions'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'looking': data_raw['looks'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        d['scene'] = self.get_scene_tags(data_raw['traffic'])
        d['tte'] = []
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])

        else:
            overlap = opts['overlap'] if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res

            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx, olap_res)])
                d[k] = seqs
            tte_seq = []
            for seq in d['box']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                tte_seq.extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
            d['tte'] = tte_seq
        for k in d.keys():
            d[k] = np.array(d[k])

        if opts['pred_target_type'] != 'scene':
            if opts['pred_target_type'] == 'crossing':
                dcount = d[opts['pred_target_type']][:, 0, :]
            else:
                dcount = d[opts['pred_target_type']].reshape((-1, 1))
            pos_count = np.count_nonzero(dcount)
            neg_count = len(dcount) - pos_count
            print("{} : Negative {} and positive {} sample counts".format(opts['pred_target_type'],
                                                                          neg_count, pos_count))
        else:
            pos_count, neg_count = 0, 0
        return d, neg_count, pos_count

    def get_data(self, data_type, data_raw, model_opts):
        model_opts['normalize_boxes'] = False
        process = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        dataset = model_opts['dataset']
        self._generator = model_opts.get('generator', False)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        _data_samples = {}
        data_type_sizes_dict = {}

        # crop only bounding boxes
        if 'ped_legs' in model_opts['obs_input_type']:
            print('\n#####################################')
            print('Generating pedestrian leg samples {}'.format(data_type))
            print('#####################################')
            path_to_ped_legs, _ = get_path(save_folder='_'.join(['ped_legs', aux_name]),
                                           dataset=dataset,
                                           save_root_folder='data/features')
            leg_coords = np.copy(data['box_org'])
            for seq in leg_coords:
                for bbox in seq:
                    height = bbox[3] - bbox[1]
                    bbox[1] = bbox[1] + height // 2
            target_dim = (227, 227) if self._backbone == 'alexnet' else (224,224)
            data['ped_legs'], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                 leg_coords,
                                                                 data['ped_id'],
                                                                 data_type=data_type,
                                                                 save_path=path_to_ped_legs,
                                                                 crop_type='bbox',
                                                                 crop_mode='warp',
                                                                 target_dim=target_dim,
                                                                 process=process)

            data_type_sizes_dict['ped_legs'] = feat_shape
        if 'ped_head' in model_opts['obs_input_type']:
            print('\n#####################################')
            print('Generating pedestrian head samples {}'.format(data_type))
            print('#####################################')

            path_to_ped_heads, _ = get_path(save_folder='_'.join(['ped_head', aux_name]),
                                              dataset=dataset,
                                              save_root_folder='data/features')
            head_coords = np.copy(data['box_org'])
            for seq in head_coords:
                for bbox in seq:
                    height = bbox[3] - bbox[1]
                    bbox[3] = bbox[3] - (height * 2) // 3
            target_dim = (227, 227) if self._backbone == 'alexnet' else (224,224)
            data['ped_head'], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                 head_coords,
                                                                 data['ped_id'],
                                                                 data_type=data_type,
                                                                 save_path=path_to_ped_heads,
                                                                 crop_type='bbox',
                                                                 crop_mode='warp',
                                                                 target_dim=target_dim,
                                                                 process=process)
            data_type_sizes_dict['ped_head'] = feat_shape
        if 'scene_context' in model_opts['obs_input_type']:
            print('\n#####################################')
            print('Generating local context {}'.format(data_type))
            print('#####################################')
            target_dim = (540, 540) if self._backbone == 'alexnet' else (224, 224)
            path_to_scene_context, _ = get_path(save_folder='_'.join(['scene_context', aux_name]),
                                                dataset=dataset,
                                                save_root_folder='data/features')
            data['scene_context'], feat_shape = self.load_images_crop_and_process(data['image'],
                                                                      data['box_org'],
                                                                      data['ped_id'],
                                                                      data_type=data_type,
                                                                      save_path=path_to_scene_context,
                                                                      crop_type='none',
                                                                      target_dim=target_dim,
                                                                      process=process)
            data_type_sizes_dict['scene_context'] = feat_shape

        # Reshape the sample tracks by collapsing sequence size to the number of samples
        # (samples, seq_size, features) -> (samples*seq_size, features)
        if model_opts.get('reshape', False):
            for k in data:
                dsize = data_type_sizes_dict.get(k, data[k].shape)
                if self._generator:
                    new_shape = (-1, data[k].shape[-1]) if data[k].ndim > 2 else (-1, 1)
                else:
                    new_shape = (-1,) + dsize[1:] if len(dsize) > 3  else (-1, dsize[-1])
                data[k] = np.reshape(data[k], new_shape)
                data_type_sizes_dict[k] = dsize[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            _data.append(data[d_type])
            data_sizes.append(data_type_sizes_dict[d_type])
            data_types.append(d_type)

        classes = 7 if model_opts['pred_target_type'] == 'scene' else 2

        # create the final data file to be returned
        if self._generator:
            is_train_data = True
            if data_type == 'test' or model_opts.get('predict_data', False):
                is_train_data = False
            data_inputs = []
            for i, d in enumerate(_data):
                data_inputs.append(DataGenerator(data=[d],
                                   labels=data[model_opts['pred_target_type']],
                                   data_sizes=[data_sizes[i]],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=[model_opts['obs_input_type'][i]],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=is_train_data,
                                   to_fit=is_train_data))
            _data = (data_inputs, data[model_opts['pred_target_type']]) # set y to None
        else:
            _data = (_data, data[model_opts['pred_target_type']])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes,
                                'pred_type': model_opts['pred_target_type'],
                                'num_classes': classes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=20,
              learning_scheduler=None,
              model_opts=None,
              **kwargs):
        model_opts['model_folder_name'] = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        model_opts['reshape'] = True

        model_opts['pred_target_type'] = 'walking'
        model_opts['obs_input_type'] = ['ped_legs']
        walk_model = self.train_model(data_train, data_val,
                                      learning_scheduler=learning_scheduler,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      model_opts=model_opts)

        model_opts['pred_target_type'] = 'looking'
        model_opts['obs_input_type'] = ['ped_head']
        look_model = self.train_model(data_train, data_val,
                                      learning_scheduler=learning_scheduler,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      model_opts=model_opts)



        model_opts['obs_input_type'] = ['scene_context']
        model_opts['pred_target_type'] = 'scene'
        scene_model = self.train_model(data_train, data_val,
                                       learning_scheduler=learning_scheduler,
                                       epochs=epochs, lr=0.000625,
                                       batch_size=batch_size,
                                       loss_func='categorical_crossentropy',
                                       activation='sigmoid',
                                       model_opts=model_opts)

        model_opts['model_paths'] = {'looking': look_model, 'walking': walk_model, 'scene': scene_model}
        saved_files_path = self.train_final(data_train, model_opts=model_opts)

        return saved_files_path

    def train_model(self, data_train,
                    data_val=None, batch_size=32,
                    epochs=60, lr=0.00001,
                    optimizer='sgd',
                    loss_func='sparse_categorical_crossentropy',
                    activation='sigmoid',
                    learning_scheduler =None,
                    model_opts=None):
        """
        Trains a single model
        Args:
            data_train: Training data
            data_val: Validation data
            loss_func: The type of loss function to use
            activation: The activation type for the last (predictions) layer
            For other parameters refer to train()
        """
        learning_scheduler = learning_scheduler or {}

        # Set the path for saving models
        model_folder_name = model_opts.get('model_folder_name',
                                           time.strftime("%d%b%Y-%Hh%Mm%Ss"))
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset'],
                       'sub_folder': model_opts['pred_target_type']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})
        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0][0]

        # Create model
        data_train['data_params']['activation'] = activation
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        if data_train['data_params']['num_classes'] > 2:
            model_opts['apply_class_weights'] = False
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        callbacks = self.get_callbacks(learning_scheduler, model_path)

        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        history = train_model.fit(x=data_train['data'][0][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  callbacks=callbacks,
                                  verbose=2)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def train_final(self, data_train,
                    batch_size=1,
                    num_iterations=1000,
                    model_opts=None):
        """
        Trains the final SVM model
        Args:
            data_train: Training data
            batch_size: Batch sizes used for generator
            num_iterations: Number of iterations for SVM model
            model_opts: Model options
        Returns:
            The path to the root folder where models are saved
        """
        print("Training final model!")
        # Set the path for saving models
        model_folder_name = model_opts.get('model_folder_name',
                                           time.strftime("%d%b%Y-%Hh%Mm%Ss"))
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}

        # Read train data
        model_opts['obs_input_type'] = ['ped_head', 'ped_legs', 'scene_context']
        model_opts['pred_target_type'] = 'crossing'

        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size,
                                                         'predict_data': True})

        if data_train['data_params']['num_classes'] > 2:
            model_opts['apply_class_weights'] = False
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])

        # Load conv model
        look_model = self.get_model({'data_sizes': [data_train['data_params']['data_sizes'][0]],
                                     'weights': model_opts['model_paths']['looking'], 'features': True})
        looking_features = look_model.predict(data_train['data'][0][0], verbose=1)

        walk_model = self.get_model({'data_sizes': [data_train['data_params']['data_sizes'][1]],
                                     'weights': model_opts['model_paths']['walking'], 'features': True})
        walking_features = walk_model.predict(data_train['data'][0][1], verbose=1)

        scene_model = self.get_model({'data_sizes': [data_train['data_params']['data_sizes'][2]],
                                     'weights': model_opts['model_paths']['scene'], 'features': True,
                                      'num_classes': 7})
        scene_features = scene_model.predict(data_train['data'][0][2],  verbose=1)

        svm_features = np.concatenate([looking_features, walking_features, scene_features], axis=-1)

        svm_model = make_pipeline(StandardScaler(),
                                  LinearSVC(random_state=0, tol=1e-5,
                                            max_iter=num_iterations,
                                            class_weight=class_w))
        svm_model.fit(svm_features, np.squeeze(data_train['data'][1]))

        # Save configs
        model_path, saved_files_path = get_path(**path_params, file_name='model.pkl')
        with open(model_path, 'wb') as fid:
            pickle.dump(svm_model, fid, pickle.HIGHEST_PROTOCOL)

        # Save configs
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path


    def get_model(self, data_params, *args, **kwargs):
        K.clear_session()
        net_model = self._conv_models[self._backbone](input_shape=data_params['data_sizes'][0],
                            include_top=False, weights=self._weights)

        # Convert to fully connected
        net_model = convert_to_fcn(net_model, classes=data_params.get('num_classes', 2),
                       activation=data_params.get('activation', 'softmax'),
                       pooling=self._pooling, features=data_params.get('features', False))
        net_model.summary()
        return net_model


    def test(self, data_test, model_path=''):
        """
        Test function
        :param data_test: The raw data received from the dataset interface
        :param model_path: The path to the folder where the model and config files are saved.
        :return: The following performance metrics: acc, auc, f1, precision, recall
        """
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        data_test = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})

        # Load conv model
        look_model = self.get_model({'data_sizes': [data_test['data_params']['data_sizes'][0]],
                                     'weights': model_opts['model_paths']['looking'], 'features': True})
        walk_model = self.get_model({'data_sizes': [data_test['data_params']['data_sizes'][1]],
                                     'weights': model_opts['model_paths']['walking'], 'features': True})
        scene_model = self.get_model({'data_sizes': [data_test['data_params']['data_sizes'][2]],
                                      'weights': model_opts['model_paths']['scene'], 'features': True,
                                      'num_classes': 7})

        with open(os.path.join(model_path, 'model.pkl'), 'rb') as fid:
            try:
                svm_model = pickle.load(fid)
            except:
                svm_model = pickle.load(fid, encoding='bytes')

        looking_features = look_model.predict(data_test['data'][0][0], verbose=1)
        walking_features = walk_model.predict(data_test['data'][0][1], verbose=1)
        scene_features = scene_model.predict(data_test['data'][0][2], verbose=1)
        svm_features = np.concatenate([looking_features, walking_features, scene_features], axis=-1)
        res = svm_model.predict(svm_features)
        res = np.reshape(res, (-1, model_opts['obs_length'], 1))
        results = np.mean(res, axis=1)

        gt = np.reshape(data_test['data'][1], (-1, model_opts['obs_length'], 1))[:, 1, :]
        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        data_tte = np.squeeze(data_test['tte'][:len(gt)])

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                          recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        with open(save_results_path, 'w') as fid:
            yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


class TwoStream(ActionPredict):
    """
    This is an implementation of two-stream network based on
    Simonyan et al. "Two-stream convolutional networks for action recognition
    in videos.", NeurIPS, 2014.
    """
    def __init__(self,
                 dropout=0.0,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='imagenet',
                 backbone='vgg16',
                 num_classes=1,
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
            num_classes: Number of activity classes to predict.
            backbone: Backbone network. Only vgg16 is supported.
        """
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._num_classes = num_classes
        if backbone != 'vgg16':
            print("Only vgg16 backbone is supported")
            backbone ='vgg16'
        self._backbone = backbone
        self._conv_model = vgg16.VGG16

    def get_data_sequence(self, data_type, data_raw, opts):

        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'box': data_raw['bbox'].copy(),
             'crossing': data_raw['activities'].copy(),
             'walking': data_raw['actions'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'looking': data_raw['looks'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res

            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs
            for seq in d['box']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
        for k in d.keys():
            d[k] = np.array(d[k])

        dcount = d['crossing'][:, 0, :]
        pos_count = np.count_nonzero(dcount)
        neg_count = len(dcount) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def get_data(self, data_type, data_raw, model_opts):
        model_opts['normalize_boxes'] = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        process = False
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']
        self._generator = model_opts.get('generator', False)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        feature_type = model_opts['obs_input_type'][0]

        # Only 3 types of rgb features are supported
        assert feature_type in ['local_box', 'local_context', 'scene']

        _data_samples = {'crossing': data['crossing']}
        data_type_sizes_dict = {}

        data_gen_params = {'data_type': data_type, 'crop_type': 'none'}

        if feature_type == 'local_box':
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif feature_type == 'local_context':
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio

        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join([feature_type, aux_name, str(eratio)]) \
                           if feature_type in ['local_context', 'local_surround'] \
                           else '_'.join([feature_type, aux_name])
        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')
        data_gen_params['save_path'] = path_to_features

        # Extract relevant frames based on the optical flow length
        ofl = model_opts.get('optical_flow_length', 10)
        stidx = ofl - round((ofl + 1) / 2)
        endidx = (ofl + 1) // 2

        # data_type_sizes_dict[feature_type] = (_data_samples[feature_type].shape[1], *feat_shape[1:])
        _data_samples['crossing'] = _data_samples['crossing'][:, stidx:-endidx, ...]
        effective_dimension = _data_samples['crossing'].shape[1]

        _data_samples[feature_type], feat_shape = self.load_images_crop_and_process(data['image'][:, stidx:-endidx, ...],
                                                                                    data['box_org'][:, stidx:-endidx, ...],
                                                                                    data['ped_id'][:, stidx:-endidx, ...],
                                                                                    process=process,
                                                                                    **data_gen_params)
        data_type_sizes_dict[feature_type] = feat_shape

        print('\n#####################################')
        print('Generating optical flow {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join([feature_type, 'flow', str(eratio)]) \
                           if feature_type == 'local_context' else '_'.join([feature_type, 'flow'])
        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')
        data_gen_params['save_path'] = path_to_features
        _data_samples['optical_flow'], feat_shape = self.get_optical_flow(data['image'],
                                                                          data['box_org'],
                                                                          data['ped_id'],
                                                                          **data_gen_params)

        # Create opflow data by stacking batches of optflow
        opt_flow = []
        if self._generator:
            _data_samples['optical_flow'] = np.expand_dims(_data_samples['optical_flow'], axis=-1)

        for sample in _data_samples['optical_flow']:
            opf = [np.concatenate(sample[i:i+ofl, ...], axis=-1) for i in range(sample.shape[0] - ofl + 1)]
            opt_flow.append(opf)
        _data_samples['optical_flow'] = np.array(opt_flow)
        if self._generator:
            data_type_sizes_dict['optical_flow'] = (feat_shape[0] - ofl + 1,
                                                    *feat_shape[1:3], feat_shape[3]*ofl)
        else:
            data_type_sizes_dict['optical_flow'] = _data_samples['optical_flow'].shape[1:]

        if model_opts.get('reshape', False):
            for k in _data_samples:
                dsize = data_type_sizes_dict.get(k, _data_samples[k].shape)
                if self._generator:
                    new_shape = (-1, _data_samples[k].shape[-1]) if _data_samples[k].ndim > 2 else (-1, 1)
                else:
                    new_shape = (-1,) + dsize[1:] if len(dsize) > 3 else (-1, dsize[-1])
                _data_samples[k] = np.reshape(_data_samples[k], new_shape)
                data_type_sizes_dict[k] = dsize[1:]
        # create the final data file to be returned
        if self._generator:
            _data_rgb = (DataGenerator(data=[_data_samples[feature_type]],
                                   labels=_data_samples['crossing'],
                                   data_sizes=[data_type_sizes_dict[feature_type]],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), _data_samples['crossing'])  # set y to None
            _data_opt_flow = (DataGenerator(data=[_data_samples['optical_flow']],
                                   labels=_data_samples['crossing'],
                                   data_sizes=[data_type_sizes_dict['optical_flow']],
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=['optical_flow'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test',
                                   stack_feats=True), _data_samples['crossing'])  # set y to None
        else:
            _data_rgb = (_data_samples[feature_type], _data_samples['crossing'])
            _data_opt_flow = (_data_samples['optical_flow'], _data_samples['crossing'])

        return {'data_rgb': _data_rgb,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'data_opt_flow': _data_opt_flow,
                'data_params_rgb': {'data_types': [feature_type],
                                    'data_sizes': [data_type_sizes_dict[feature_type]]},
                'data_params_opt_flow': {'data_types': ['optical_flow'],
                                         'data_sizes': [data_type_sizes_dict['optical_flow']]},
                'effective_dimension': effective_dimension,
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def add_dropout(self, model, add_new_pred=False):
        """
           Adds dropout layers to a given vgg16 network. If specified, changes the dimension of
           the last layer (predictions)
           Args:
               model: A given vgg16 model
               add_new_pred: Whether to change the final layer
           Returns:
               Returns the new model
        """
        # Change to a single class output and add dropout
        fc1_dropout = Dropout(self._dropout)(model.layers[-3].output)
        fc2 = model.layers[-2](fc1_dropout)
        fc2_dropout = Dropout(self._dropout)(fc2)
        if add_new_pred:
            output = Dense(self._num_classes, name='predictions', activation='sigmoid')(fc2_dropout)
        else:
            output = model.layers[-1](fc2_dropout)

        return Model(inputs=model.input, outputs=output)

    def get_model(self, data_params, *args, **kwargs):
        K.clear_session()
        data_size = data_params['data_sizes'][0]
        net_model = self._conv_model(input_shape=data_size,
                                     include_top=True, weights=self._weights)
        net_model = self.add_dropout(net_model, add_new_pred=True)

        if self._freeze_conv_layers and self._weights:
            for layer in net_model.layers:
                if 'conv' in layer.name:
                    layer.trainable = False
        net_model.summary()
        return net_model

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              optimizer='sgd',
              learning_scheduler=None,
              model_opts=None):

        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_opts['reshape'] = True
        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})
        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])

        # Get a copy of local parameters in the function minus self parameter
        local_params = {k: v for k, v in locals().items() if k != 'self'}

        #####  Optical flow model
        # Flow data shape: (1, num_frames, 224, 224, 2)
        self.train_model(model_type='opt_flow', **local_params)

        ##### rgb model
        self.train_model(model_type='rgb', **local_params)

        # Save settings
        model_opts_path, saved_files_path = get_path(**path_params,
                                                     file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        return saved_files_path


    def train_model(self, model_type, data_train, data_val,
                    class_w, learning_scheduler, path_params, optimizer,
                    batch_size, epochs, lr, **kwargs):
        """
        Trains a single model
        Args:
            train_data: Training data
            val_data: Validation data
            model_type: The model type, 'rgb' or 'opt_flow'
            path_params: Parameters for generating paths for saving models and configurations
            callbacks: List of training call back functions
            class_w: Class weights
            For other parameters refer to train()
        """
        learning_scheduler = learning_scheduler or {}
        if model_type == 'opt_flow':
            self._weights = None
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model = self.get_model(data_train['data_params_' + model_type])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        data_path_params = {**path_params, 'sub_folder': model_type}
        model_path, _ = get_path(**data_path_params, file_name='model.h5')
        callbacks = self.get_callbacks(learning_scheduler,model_path)

        if data_val:
            data_val = data_val['data_' + model_type]
            if self._generator:
                data_val = data_val[0]

        history = train_model.fit(x=data_train['data_' + model_type][0],
                                  y=None if self._generator else data_train['data_' + model_type][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)

        if 'checkpoint' not in learning_scheduler:
            print('{} train model is saved to {}'.format(model_type, model_path))
            train_model.save(model_path)
        # Save training history
        history_path, saved_files_path = get_path(**data_path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

    def test(self, data_test, model_path=''):

        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        test_data = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
        rgb_model = load_model(os.path.join(model_path, 'rgb', 'model.h5'))
        opt_flow_model = load_model(os.path.join(model_path, 'opt_flow', 'model.h5'))

        # Evaluate rgb model
        results_rgb = rgb_model.predict(test_data['data_rgb'][0], verbose=1)
        results_rgb = np.reshape(results_rgb, (-1, test_data['effective_dimension'], 1))
        results_rgb = np.mean(results_rgb, axis=1)

        # Evaluate optical flow model
        results_opt_flow = opt_flow_model.predict(test_data['data_opt_flow'][0], verbose=1)
        results_opt_flow = np.reshape(results_opt_flow, (-1, test_data['effective_dimension'], 1))
        results_opt_flow = np.mean(results_opt_flow, axis=1)

        # Average the predictions for both streams
        results = (results_rgb + results_opt_flow) / 2.0

        gt = np.reshape(test_data['data_rgb'][1], (-1, test_data['effective_dimension'], 1))[:, 1, :]

        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                          recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        with open(save_results_path, 'w') as fid:
            yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


class TwoStreamFusion(ActionPredict):
    """
    This is an implementation of two-stream network with fusion mechanisms based
    on Feichtenhofer, Christoph et al. "Convolutional two-stream network fusion for
     video action recognition." CVPR, 2016.
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=True,
                 weights='imagenet',
                 fusion_point='early', # early, late, two-stage
                 fusion_method='sum',
                 num_classes=1,
                 backbone='vgg16',
                 **kwargs):
        """
        Class init function
        Args:
            dropout: Dropout value for fc6-7 of vgg16.
            dense_activation: Activation of last dense (predictions) layer.
            freeze_conv_layers: If set true, only fc layers of the networks are trained
            weights: Pre-trained weights for networks.
            fusion_point: At what point the networks are fused (for details refer to the paper).
            Options are: 'early' (streams are fused after block 4),'late' (before the loss layer),
            'two-stage' (streams are fused after block 5 and before loss).
            fusion_method: How the weights of fused layers are combined.
            Options are: 'sum' (weights are summed), 'conv' (weights are concatenated and fed into
            a 1x1 conv to reduce dimensions to the original size).
            num_classes: Number of activity classes to predict.
            backbone: Backbone network. Only vgg16 is supported.
       """
        super().__init__(**kwargs)
        # Network parameters
        assert fusion_point in ['early', 'late', 'two-stage'], \
        "fusion point {} is not supported".format(fusion_point)

        assert fusion_method in ['sum', 'conv'], \
        "fusion method {} is not supported".format(fusion_method)

        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._num_classes = num_classes
        if backbone != 'vgg16':
            print("Only vgg16 backbone is supported")
        self._conv_models = vgg16.VGG16
        self._fusion_point = fusion_point
        self._fusion_method = fusion_method

    def get_data_sequence(self, data_type, data_raw, opts):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'box': data_raw['bbox'].copy(),
             'crossing': data_raw['activities'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']

        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])

        else:
            overlap = opts['overlap'] if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res

            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs
            for seq in d['box']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                range(start_idx, end_idx + 1, olap_res)])
        for k in d.keys():
            d[k] = np.array(d[k])

        dcount = d['crossing'][:, 0, :]
        pos_count = np.count_nonzero(dcount)
        neg_count = len(dcount) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def get_data(self, data_type, data_raw, model_opts):
        model_opts['normalize_boxes'] = False
        process = False
        aux_name = '_'.join([self._backbone, 'raw']).strip('_')
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']
        self._generator = model_opts.get('generator', False)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        feature_type = model_opts['obs_input_type'][0]

        # Only 3 types of rgb features are supported
        assert feature_type in ['local_box', 'local_context', 'scene']

        _data_samples = {'crossing': data['crossing']}
        data_type_sizes_dict = {}
        data_gen_params = {'data_type': data_type, 'crop_type': 'none'}

        if feature_type == 'local_box':
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif feature_type == 'local_context':
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio

        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')

        save_folder_name = '_'.join([feature_type, aux_name, str(eratio)]) \
                           if feature_type in ['local_context', 'local_surround'] \
                           else '_'.join([feature_type, aux_name])

        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')
        data_gen_params['save_path'] = path_to_features

        # Extract relevant rgb frames based on the optical flow length
        # Optical flow length is either 5 or 10. For example, for length of 10, and
        # sequence size of 16, 7 rgb frames are selected.
        ofl = model_opts['optical_flow_length']
        stidx = ofl - round((ofl + 1) / 2)
        endidx = (ofl + 1) // 2

        _data_samples['crossing'] = _data_samples['crossing'][:, stidx:-endidx, ...]
        effective_dimension = _data_samples['crossing'].shape[1]

        _data_samples[feature_type], feat_shape = self.load_images_crop_and_process(data['image'][:, stidx:-endidx, ...],
                                                                                    data['box_org'][:, stidx:-endidx, ...],
                                                                                    data['ped_id'][:, stidx:-endidx, ...],
                                                                                    process=process,
                                                                                    **data_gen_params)
        data_type_sizes_dict[feature_type] = feat_shape


        print('\n#####################################')
        print('Generating {} optical flow {}'.format(feature_type, data_type))
        print('#####################################')
        save_folder_name = '_'.join([feature_type, 'flow',  str(eratio)]) \
                                    if feature_type in ['local_context', 'local_surround'] \
                                    else '_'.join([feature_type, 'flow'])

        path_to_features, _ = get_path(save_folder=save_folder_name,
                                       dataset=dataset,
                                       save_root_folder='data/features')

        data_gen_params['save_path'] = path_to_features
        _data_samples['optical_flow'], feat_shape = self.get_optical_flow(data['image'],
                                                                          data['box_org'],
                                                                          data['ped_id'],
                                                                          **data_gen_params)

        # Create opflow data by stacking batches of optflow
        opt_flow = []
        if self._generator:
            _data_samples['optical_flow'] = np.expand_dims(_data_samples['optical_flow'], axis=-1)

        for sample in _data_samples['optical_flow']:
            opf = [np.concatenate(sample[i:i + ofl, ...], axis=-1) for i in range(sample.shape[0] - ofl + 1)]
            opt_flow.append(opf)
        _data_samples['optical_flow'] = np.array(opt_flow)
        if self._generator:
            data_type_sizes_dict['optical_flow'] = (feat_shape[0] - ofl + 1,
                                                    *feat_shape[1:3], feat_shape[3] * ofl)
        else:
            data_type_sizes_dict['optical_flow'] = _data_samples['optical_flow'].shape[1:]

        if model_opts.get('reshape', False):
            for k in _data_samples:
                dsize = data_type_sizes_dict.get(k, _data_samples[k].shape)
                if self._generator:
                    new_shape = (-1, _data_samples[k].shape[-1]) if _data_samples[k].ndim > 2 else (-1, 1)
                else:
                    new_shape = (-1,) + dsize[1:] if len(dsize) > 3 else (-1, dsize[-1])
                _data_samples[k] = np.reshape(_data_samples[k], new_shape)
                data_type_sizes_dict[k] = dsize[1:]

        _data = [_data_samples[feature_type], _data_samples['optical_flow']]
        data_sizes = [data_type_sizes_dict[feature_type],
                      data_type_sizes_dict['optical_flow']]
        data_types = [feature_type, 'optical_flow']

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=_data_samples['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=[feature_type,'optical_flow'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test',
                                   stack_feats=True),
                                   _data_samples['crossing'])
        else:
            _data = (_data, _data_samples['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types,
                                'data_sizes': data_sizes},
                'effective_dimension': effective_dimension,
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def fuse_layers(self, l1, l2):
        """
        Fuses two given layers (tensors)
        Args:
            l1:  First tensor
            l2:  Second tensor
        Returns:
            Fused tensors based on a given method.
        """

        if self._fusion_method == 'sum':
            return Add()([l1, l2])
        elif self._fusion_method == 'conv':
            concat_layer = Concatenate()([l1, l2])
            return Conv2D(l2.shape[-1], 1, 1)(concat_layer)

    def add_dropout(self, model, add_new_pred = False):
        """
        Adds dropout layers to a given vgg16 network. If specified, changes the dimension of
        the last layer (predictions)
        Args:
            model: A given vgg16 model
            add_new_pred: Whether to change the final layer
        Returns:
            Returns the new model
        """

        # Change to a single class output and add dropout
        fc1_dropout = Dropout(self._dropout)(model.layers[-3].output)
        fc2 = model.layers[-2](fc1_dropout)
        fc2_dropout = Dropout(self._dropout)(fc2)
        if add_new_pred:
            output = Dense(self._num_classes, name='predictions', activation='sigmoid')(fc2_dropout)
        else:
            output = model.layers[-1](fc2_dropout)

        return Model(inputs=model.input, outputs=output)

    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              optimizer='sgd',
              learning_scheduler=None,
              model_opts=None):
        learning_scheduler = learning_scheduler or {}

        # Generate parameters for saving models and configurations
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")

        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        model_opts['reshape'] = True
        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})
            data_val = data_val['data']
            if self._generator:
                data_val = data_val[0]

        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model = self.get_model(data_train['data_params'])
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        callbacks = self.get_callbacks(learning_scheduler, model_path)

        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)
        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        # Save settings
        model_opts_path, saved_files_path = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        return saved_files_path

    def get_model(self, data_params, *args, **kwargs):
        data_size = data_params['data_sizes'][0]
        rgb_model = self._conv_models(input_shape=data_size,
                                          include_top=True, weights=self._weights)
        rgb_model = self.add_dropout(rgb_model, add_new_pred=True)
        data_size = data_params['data_sizes'][1]
        temporal_model = self._conv_models(input_shape=data_size,
                                           include_top=True, weights=None, classes=1)
        temporal_model = self.add_dropout(temporal_model)

        for layer in rgb_model.layers:
            layer._name = "rgb_" + layer._name
            if self._freeze_conv_layers and 'conv' in layer._name:
                layer.trainable = False

        # rgb_model.load_weights('')
        if self._fusion_point == 'late':
            output = Average()([temporal_model.output, rgb_model.output])

        if self._fusion_point == 'early':
            fusion_point = 'block4_pool'
            rgb_fuse_layer = rgb_model.get_layer('rgb_' + fusion_point).output
            start_fusion = False
            for layer in temporal_model.layers:
                if layer.name == fusion_point:
                    x = self.fuse_layers(rgb_fuse_layer, layer.output)
                    start_fusion = True
                    continue
                if start_fusion:
                    x = layer(x)
                else:
                   layer.trainable = False

            output = x

        if self._fusion_point == 'two-stage':
            fusion_point = 'block5_conv3'
            rgb_fuse_layer = rgb_model.get_layer('rgb_' + fusion_point).output
            start_fusion = False
            for layer in temporal_model.layers:
                if layer.name == fusion_point:
                    x = self.fuse_layers(rgb_fuse_layer, layer.output)
                    start_fusion = True
                    continue
                if start_fusion:
                    x = layer(x)
                else:
                   layer.trainable = False

            output = Average()([x, rgb_model.output])

        net_model = Model(inputs=[rgb_model.input, temporal_model.input],
                          outputs=output)
        plot_model(net_model, to_file='model.png',
                   show_shapes=False, show_layer_names=False,
                   rankdir='TB', expand_nested=False, dpi=96)

        net_model.summary()
        return net_model

    def test(self, data_test, model_path=''):
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        data_test = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})

        # Load conv model
        test_model = load_model(os.path.join(model_path, 'model.h5'))
        results = test_model.predict(data_test['data'][0], batch_size=8, verbose=1)
        results = np.reshape(results, (-1, data_test['effective_dimension'], 1))
        results = np.mean(results, axis=1)

        gt = np.reshape(data_test['data'][1], (-1, data_test['effective_dimension'], 1))[:, 1, :]
        acc = accuracy_score(gt, np.round(results))
        f1 = f1_score(gt, np.round(results))
        auc = roc_auc_score(gt, np.round(results))
        roc = roc_curve(gt, results)
        precision = precision_score(gt, np.round(results))
        recall = recall_score(gt, np.round(results))
        pre_recall = precision_recall_curve(gt, results)

        print('acc:{:.2f} auc:{:0.2f} f1:{:0.2f} precision:{:0.2f} recall:{:0.2f}'.format(acc, auc, f1, precision,
                                                                                  recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

        with open(save_results_path, 'w') as fid:
            yaml.dump(results, fid)
        return acc, auc, f1, precision, recall


def attention_3d_block(hidden_states, dense_size=128, modality=''):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec'+modality)(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state'+modality)(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score'+modality)
    attention_weights = Activation('softmax', name='attention_weight'+modality)(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector'+modality)
    pre_activation = concatenate([context_vector, h_t], name='attention_output'+modality)
    attention_vector = Dense(dense_size, use_bias=False, activation='tanh', name='attention_vector'+modality)(pre_activation)
    return attention_vector


class PCPA(ActionPredict):

    """
    Hybridization of MultiRNN with 3D convolutional features and attention

    many-to-one attention block is adapted from:
    https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py

    """
    
    def __init__(self,
                 num_hidden_units=256,
                 cell_type='gru',
                 **kwargs):
        """
        Class init function
        
        Args:
            num_hidden_units: Number of recurrent hidden layers
            cell_type: Type of RNN cell
            **kwargs: Description
        """
        super().__init__(**kwargs)
        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._rnn = self._gru if cell_type == 'gru' else self._lstm
        self._rnn_cell = GRUCell if cell_type == 'gru' else LSTMCell
        assert self._backbone in ['c3d', 'i3d'], 'Incorrect backbone {}! Should be C3D or I3D'.format(self._backbone)
        self._3dconv = C3DNet if self._backbone == 'c3d' else I3DNet

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        model_opts_3d = model_opts.copy()

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type:
                if self._backbone == 'c3d':
                    model_opts_3d['target_dim'] = (112, 112)
                model_opts_3d['process'] = False
                features, feat_shape = self.get_context_data(model_opts_3d, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features, _ = get_pose(model_opts,
                                    data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'tte': data['tte'],
                'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_model(self, data_params, *args, **kwargs):
        return_sequence = True
        data_sizes = data_params['data_sizes']
        data_types = data_params['data_types']
        network_inputs = []
        encoder_outputs = []
        core_size = len(data_sizes)

        conv3d_model = self._3dconv()
        network_inputs.append(conv3d_model.input)

        attention_size = self._num_hidden_units

        if self._backbone == 'i3d':
            x = Flatten(name='flatten_output')(conv3d_model.output)
            x = Dense(name='emb_'+self._backbone,
                       units=attention_size,
                       activation='sigmoid')(x)
        else:
            x = conv3d_model.output
            x = Dense(name='emb_'+self._backbone,
                       units=attention_size,
                       activation='sigmoid')(x)

        encoder_outputs.append(x)

        for i in range(1, core_size):
            network_inputs.append(Input(shape=data_sizes[i], name='input_' + data_types[i]))
            encoder_outputs.append(self._rnn(name='enc_' + data_types[i], r_sequence=return_sequence)(network_inputs[i]))

        if len(encoder_outputs) > 1:
            att_enc_out = []
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(encoder_outputs[0])
            att_enc_out.append(x) # first output is from 3d conv netwrok 
            # for recurrent branches apply many-to-one attention block
            for i, enc_out in enumerate(encoder_outputs[1:]):
                x = attention_3d_block(enc_out, dense_size=attention_size, modality='_'+data_types[i])
                x = Dropout(0.5)(x)
                x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
                att_enc_out.append(x)
            # aplly many-to-one attention block to the attended modalities
            x = Concatenate(name='concat_modalities', axis=1)(att_enc_out)
            encodings = attention_3d_block(x, dense_size=attention_size, modality='_modality')

            #print(encodings.shape)
            #print(weights_softmax.shape)
        else:
            encodings = encoder_outputs[0]

        model_output = Dense(1, activation='sigmoid',
                             name='output_dense',
                             activity_regularizer=regularizers.l2(0.001))(encodings)

        net_model = Model(inputs=network_inputs,
                          outputs=model_output)
        net_model.summary()
        #plot_model(net_model, to_file='MultiRNN3D_ATT.png')
        return net_model

class VideoMAE(ActionPredict):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
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

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super(VideoMAE, self).get_data(data_type, data_raw, model_opts)

    # TODO: use keras function to load weights
    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        if not os.path.exists(self._weights):
            weights_url = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'
            wget.download(weights_url, self._weights)
        net_model = C3DNet(freeze_conv_layers=self._freeze_conv_layers,
                           dropout=self._dropout,
                           dense_activation=self._dense_activation,
                           include_top=True,
                           weights=self._weights)
        net_model.summary()

        return net_model

class Timesformer(ActionPredict):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
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

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super(Timesformer, self).get_data(data_type, data_raw, model_opts)

    # TODO: use keras function to load weights
    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        """
        if not os.path.exists(self._weights):
            weights_url = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'
            wget.download(weights_url, self._weights)
        net_model = C3DNet(freeze_conv_layers=self._freeze_conv_layers,
                           dropout=self._dropout,
                           dense_activation=self._dense_activation,
                           include_top=True,
                           weights=self._weights)
        net_model.summary()
        """

        return None
    
class VisionTransformer(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super().get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class VAN(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        data = super().get_data(data_type, data_raw, model_opts)
        return data

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class SmallVAN(VAN, Static):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
class HF_Resnet50(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super().get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class SwinV2(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super().get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class ConvNextV2(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super().get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
    
class BEiT(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super().get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None

class ConvNext_VAN(Static):
    """ Copied C3D model
        TODO: some things might need to be changed
    """

    def __init__(self,
                 dropout=0.5,
                 dense_activation='sigmoid',
                 freeze_conv_layers=False,
                 weights='weights/c3d_sports1M_weights_tf.h5',
                 **kwargs):
        super().__init__(**kwargs)
        # Network parameters
        self._dropout = dropout
        self._dense_activation = dense_activation
        self._freeze_conv_layers = freeze_conv_layers
        self._weights = weights
        self._backbone = 'c3d'

    def get_data(self, data_type, data_raw, model_opts):

        assert len(model_opts['obs_input_type']) == 1
        # assert model_opts['obs_length'] == 16

        model_opts['normalize_boxes'] = False
        model_opts['target_dim'] = (224, 224)
        model_opts['process'] = False
        model_opts['backbone'] = 'c3d'
        return super().get_data(data_type, data_raw, model_opts)

    def get_model(self, data_params, *args, **kwargs):
        os.makedirs(os.path.dirname(self._weights), exist_ok=True)
        return None
