from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, concatenate, Dropout, GaussianDropout,
    AlphaDropout, BatchNormalization
)
from tensorflow.keras.activations import softmax
from tensorflow.keras import backend as K


def apply_dropout(inp, rate, dropout_type='standard', name=None):
    '''Helper function to add a dropout layer of a specified type to a model

    Parameters:
    ----------
    inp: tensor
        The input tensor
    rate: float
        The rate parameter of the dropout (proportion of units dropped)
    dropout_type: str
        The type of the dropout. Allowed values are ['standard', 'gaussian', 'alpha', 'none'], which respectively
        correspond to the Dropout, GaussianDropout, and AlphaDropout keras layers, or no dropout. The default is
        'standard'
    name: str
        This string is passed as the name parameter when constructing the layer

    Returns:
    -------
    tensor
        The output tensor after application of the dropout layer
    '''

    if dropout_type == 'standard':
        output = Dropout(rate, name=name)(inp)
    elif dropout_type == 'gaussian':
        output = GaussianDropout(rate, name=name)(inp)
    elif dropout_type == 'alpha':
        output = AlphaDropout(rate, name=name)(inp)
    elif dropout_type == 'none':
        output = inp
    else:
        raise ValueError('Unrecognised dropout type {}'.format(dropout_type))
    return output


def apply_normalization(inp, axis=-1, norm_type='batchnorm', params=None, name=None):
    '''Helper function to add a normalization layer of a specified type to a model

    Parameters:
    ----------
    inp: tensor
        The input tensor
    axis: int
        Index of the axis of the axis along which to apply the normalization, typically this will be the feature axis
    params: dict
        Dictionary of parameters passed straight into the normalization layer constructor
    norm_type: str
        The type of the normalization. Allowed values are ['none', 'batchnorm'], which respectively correspond to no
        normalization, BatchNormalization, keras/keras layers. The default is 'batchnorm'
    name: str
        This string is passed as the name parameter when constructing the layer

    Returns:
    -------
    tensor
        The output tensor after application of the normalization layer
    '''

    if params is None:
        params = {}

    if norm_type == 'none':
        return inp
    elif norm_type == 'batchnorm':
        output = BatchNormalization(name=name, axis=axis, **params)(inp)
    else:
        raise ValueError('Unrecognised normalization type ' + norm_type)

    return output


def get_unet_2D(num_classes,
                input_shape,
                kernel_size=(3, 3),
                num_convs=2,
                pool_size=(2, 2),
                activation='relu',
                output_activation='softmax',
                compression_channels=(32, 64, 128, 256, 512),
                decompression_channels=(256, 128, 64, 32),
                compression_dropout=None,
                decompression_dropout=None,
                dropout_type='standard',
                norm_type='batchnorm',
                conv_kwargs=None,
                norm_kwargs=None):
    """Constructs a U-net segmentation model

    Parameters:
    -----------
    num_classes : int
        Number of classes (including background class) in the segmentation task
    input_shape : tuple
        Shape of a single input image. (height, width, channels) or (channels, height, width)
        depending on Keras' image data format. Alternatively, height, width, and/or depth may be set to None, meaning
        that the input shape is variable along that dimension (see below).
    kernel_size : tuple
        Shape (height, width) of the convolution kernel in each conv layer
        (default 3x3)
    num_convs : int
        number of convolutional layers between each downsample/upsample (default: 2)
    pool_size : tuple
        Shape (height, width) of the pooling window used to downsample/upsample the
        image between each compression/decompression module
    activation : string or function
        The activation function used in each conv layer. May be any activation function
        recognized by Keras
    output_activation : string or function
        The activation function used immediately before the final output. May be any activation function
        recognized by Keras. Default: softmax
    compression_channels : iterable
        An iterable (e.g. list or 1D numpy array) of integers listing the number of channels
        at the output of each of the compression layers
    decompression_channels : iterable
        An iterable (e.g. list or 1D numpy array) of integers listing the number of channels
        at the output of each of the decompression layers. The length of this list must be
        one less than the length of the compression_channels list
    compression_dropout : iterable or None
        If None, no dropout is applied in the compression path. If not None, an iterable containing
        the dropout rate parameters for each compression module (or None for no dropout). The rate parameter
        represents the number of units that are dropped at train time. Must be the same length as the
        compression_channels parameter. Dropout is applied after each activation layer in the module (default: None)
    decompression_dropout : iterable or None
        If None, no dropout is applied in the decompression path. If not None, an iterable containing
        the dropout rate parameters for each decompression module (or None for no dropout). The rate parameter
        represents the number of units that are dropped at train time. Must be the same length as the
        decompression_channels parameter. Dropout is applied after each activation layer in the module (default: None)
    dropout_type : str
        Type of dropout. Choose one of 'standard', 'gaussian' or 'alpha' to apply the Dropout, Gaussian or
        AlphaDropout Keras layers respectively. Note that to create a self-normalizing network (with the
        'selu' activation), the 'alpha' dropout option should be used (default: 'standard')
    norm_type : str
        Type of normalization to apply before every activation layer, choose from '['none', 'batchnorm',
        'groupnorm', 'instancenorm'] (default: 'batchnorm').
    conv_kwargs : dict
        Allows passing arbitrary kwargs into the convolutional layers of the network (Keras Conv2D layers). Do not
        pass in values for the 'filters', 'kernel_size', 'activation', 'padding', or 'name' parameters
    norm_kwargs : dict
        Allows passing arbitrary kwargs into the normalization layers, if any, (whose type is determined by the
        'norm_type' parameter)

    Returns:
    --------
    keras.model.Model
       U-net segmentation model

    NB Not all these parameters can be chosen independently of each other. The length of the
    `compression_channels` list determines the number of times the input image is downsampled in
    the network. If the number of downsamplings is `N` then the length of this list is `N+1`
    because the final compression module is purely a convolutional layer (no downsampling/pooling).
    The image must be downsampled and upsampled the same number of times so that the output image
    is the same size as the input image. Therefore the length of the `decompression_channels` list
    (which has no final convolution) must be `N`, i.e. `len(decompression_channels)` must be
    `len(compression_channels) - 1`.

    The value of `N` (as chosen implicitly by the length of the channel lists) and the choice of
    `pool_size` together determine the valid input shapes for the image. If `pool_size` is set to
    `(p, q)`, then the input image must have shape `(H, W, C)` (or `(C, H, W)` depending on which
    convention Keras is configured to use) where `H` is a multiple of `p^N` and `W` is a multiple
    of `q^N` (`C` can be any number of channels). In this case, the image shape (height, width)
    at the narrowest point in the center of the model is `( H/(p^N) , W/(q^N) )`.

    Note that you can set the input height, width or depth parameters to 'None', which allows variable input sizes
    during training or testing. However note that the *restrictions on input size from the previous paragraph still
    apply*. Choosing None merely means that different (valid) values may be used for each batch. It is recommended that
    you specify the input value unless you need a variable input shape, as this will make debugging easier and lead to
    more understandable error messages.

    """

    if conv_kwargs is None:
        conv_kwargs = {}
    if norm_kwargs is None:
        norm_kwargs = {}

    if K.image_data_format() == "channels_first":
        h_ax, w_ax, c_ax = 2, 3, 1
    elif K.image_data_format() == "channels_last":
        h_ax, w_ax, c_ax = 1, 2, 3

    # Checks on validity of configuration options
    n_downsamplings = len(compression_channels) - 1
    net_scale_factor_h = pool_size[0] ** n_downsamplings
    net_scale_factor_w = pool_size[1] ** n_downsamplings
    if compression_dropout is None:
        compression_dropout = [None] * len(compression_channels)
    if decompression_dropout is None:
        decompression_dropout = [None] * len(decompression_channels)

    assert len(decompression_channels) == n_downsamplings, ("Length of compression_channels must be one greater than "
                                                            "length of decompression_channels")
    assert (input_shape[h_ax - 1] is None) or (input_shape[h_ax - 1] % net_scale_factor_h == 0),\
        "Input image height must be a multiple of {} (or None) based on the options provided [{}]"\
        .format(net_scale_factor_h, input_shape[h_ax - 1])
    assert (input_shape[w_ax - 1] is None) or (input_shape[w_ax - 1] % net_scale_factor_w == 0),\
        "Input image width must be a multiple of {} (or None) based on the options provided [{}]"\
        .format(net_scale_factor_w, input_shape[w_ax - 1])
    assert (len(compression_dropout) == len(compression_channels)), ("Length of compression_dropout must match length "
                                                                     "of compression_channels")
    assert (len(decompression_dropout) == len(decompression_channels)), ("Length of decompression_dropout must match "
                                                                         "length of decompression_channels")
    assert dropout_type in ['standard', 'gaussian', 'alpha'], "Unrecognized dropout type {}".format(dropout_type)

    # Certain parameters of the conv and batchnorm layers should not be overridden by user input
    controlled_conv_params = ['kernel_size', 'filters', 'activation', 'name', 'padding']
    for param in controlled_conv_params:
        assert param not in conv_kwargs, "Do not include value for parameter '{}' in conv_kwargs".format(param)
    controlled_norm_params = ['name', 'axis']
    for param in controlled_norm_params:
        assert param not in norm_kwargs, ("Do not include value for parameter '{}' in "
                                                "norm_kwargs").format(param)

    # Check on appropriateness of 'groups' parameter for groupnorm
    if norm_type == 'groupnorm':
        # The default groups value of 32 is assumed by the GroupNormalization layer if not specified by the user
        groups = norm_kwargs['groups'] if 'groups' in norm_kwargs else 32
        assert all((c % groups == 0) for c in compression_channels + decompression_channels),\
            ("When using group normlization, the number of channels in the entire network must be divisible by the "
            "groups parameter, currently set to {}. Pass a different 'groups' parameter in norm_kwargs, or change the "
            "channel configuration of the network.").format(groups)

    inputs = Input(input_shape, name='image_input')

    # Hyperparameters used for all modules
    compression_hyperparams = {'kernel_size': kernel_size, 'pool_size': pool_size, 'norm_type': norm_type,
                               'activation': activation, 'num_convs': num_convs, 'dropout_type': dropout_type,
                               'conv_kwargs': conv_kwargs, 'norm_kwargs': norm_kwargs}
    decompression_hyperparams = {'kernel_size': kernel_size, 'upsample_factor': pool_size, 'norm_type': norm_type,
                                 'activation': activation, 'num_convs': num_convs, 'dropout_type': dropout_type,
                                 'conv_kwargs': conv_kwargs, 'norm_kwargs': norm_kwargs}

    # Compression path
    compression_path = [(None, inputs)]
    for i, c, d in zip(range(len(compression_channels) - 1),
                       compression_channels[:-1],
                       compression_dropout[:-1]):
        # Each compression module takes as input the pool output of the previous module
        module_name = 'compression_' + str(i)
        compression_path.append(compression_module_2D(compression_path[-1][1], c, pool=True,
                                                      dropout_rate=d, module_name=module_name,
                                                      **compression_hyperparams))

    # Central convolution (no pooling afterwards)
    center_module = compression_module_2D(compression_path[-1][1], compression_channels[-1],
                                          pool=False, dropout_rate=compression_dropout[-1],
                                          module_name='compression_center', **compression_hyperparams)

    # Decompression path
    decompression_path = [center_module]
    for i, c, d in zip(range(len(decompression_channels)),
                       decompression_channels,
                       decompression_dropout):
        # Each decompression module takes as input the previous decompression module and the conv output of the
        # corresponding compression module
        module_name = 'decompression_' + str(i)
        decompression_path.append(decompression_module_2D(decompression_path[-1], compression_path[-(i+1)][0], c,
                                                          dropout_rate=d, module_name=module_name,
                                                          **decompression_hyperparams))

    # Final convolution and output activation
    output_conv = Conv2D(num_classes, (1, 1), padding='same', name='final_conv', **conv_kwargs)(decompression_path[-1])

    if (output_activation == 'softmax' or output_activation is softmax) and K.image_data_format() == 'channels_first':
        # This is the general case for softmax activation along an arbitrary axis
        # Required when using the channels_first convention
        # But precludes serialisation due to the lambda
        output_conv = Activation(lambda x: softmax(x, c_ax), name='segmentation_output')(output_conv)
    else:
        # The general case
        output_conv = Activation(output_activation, name='segmentation_output')(output_conv)

    model = Model(inputs=inputs, outputs=output_conv)

    return model


def compression_module_2D(inputs,
                          num_output_features,
                          pool=True,
                          kernel_size=(3, 3),
                          num_convs=2,
                          pool_size=(2, 2),
                          activation='relu',
                          module_name=None,
                          dropout_rate=None,
                          dropout_type='standard',
                          norm_type='batchnorm',
                          conv_kwargs=None,
                          norm_kwargs=None):
    '''Create a downsampling compression module

    A compression module sits on the left hand path of the model and reduces the
    size of the input image using convolutions and max pooling

    Parameters:
    -----------
    inputs : keras layer
        The input layer
    num_output_features : int
        The number of output features (dimension of the features channel at the module's output)
    pool : bool
        If true, the convolutions are followed by a max pooling to downsample.
        If false, no max pooling occurs
    kernel_size : tuple
        Shape (height, width) of the convolution kernel in each conv layer
        (default 3x3)
    num_convs : int
        number of convolutional layers before the downsample  (default: 2)
    pool_size : tuple
        Shape (height, width) of the pooling window used to downsample the image
        if 'pool' is True (default 2x2)
    activation : string or function
        The activation function used in each conv layer. May be any activation function
        recognized by Keras
    dropout_rate: None or float
        Dropout rate to apply after each activation. The dropout rate represents the fraction of units to drop at
        training time.
    dropout_type: str
        Type of dropout. Choose one of 'standard', 'gaussian' or 'alpha' to apply the Dropout, Gaussian or
        AlphaDropout Keras layers respectively. Note that to create a self-normalizing network (with the
        'selu' activation), the 'alpha' dropout option should be used.
    norm_type : str
        Type of normalization to apply before every activation layer, choose from '['none', 'batchnorm',
        'groupnorm', 'instancenorm'] (default: 'batchnorm')
    conv_kwargs : dict
        Allows passing arbitrary kwargs into the convolutional layers of the module (Keras Conv2D layers)
    norm_kwargs : dict
        Allows passing arbitrary kwargs into the normalization layers, if any, (whose type is determined by the
        'norm_type' parameter,
    module_name : str
        Name used as the basis for individual layer names within the module


    Returns:
    --------
    conv_output
        keras layer after the two convolution operations. Has same height and width as the inputs parameter, and
        a size of num_output_features along the channel dimension
    pool_output
        keras layer after the downsampling operation. Omitted if pool=False. Has width and height dimensions smaller
        than the inputs parameter (according to 'pool_size'), and size of num_output_features along the channel
        dimension
    '''

    if conv_kwargs is None:
        conv_kwargs = {}
    if norm_kwargs is None:
        norm_kwargs = {}

    if K.image_data_format() == "channels_first":
        c_ax = 1
    elif K.image_data_format() == "channels_last":
        c_ax = 3

    # Add convolutional layers
    conv_output = inputs
    for i in range(num_convs):
        # Convolutional layer
        layer_name = (module_name + '_conv_' + str(i)) if (module_name is not None) else None
        conv_output = Conv2D(num_output_features, kernel_size, padding='same',
                             name=layer_name, **conv_kwargs)(conv_output)

        # Batch norm, if required
        layer_name = (module_name + '_norm_' + str(i)) if (module_name is not None) else None
        conv_output = apply_normalization(conv_output, axis=c_ax, name=layer_name, norm_type=norm_type,
                                          params=norm_kwargs)

        # Activation
        layer_name = (module_name + '_activation_' + str(i)) if (module_name is not None) else None
        conv_output = Activation(activation, name=layer_name)(conv_output)

        # Dropout, if required
        if dropout_rate is not None:
            layer_name = (module_name + '_dropout_' + str(i)) if (module_name is not None) else None
            conv_output = apply_dropout(conv_output, dropout_rate, dropout_type=dropout_type, name=layer_name)

    # Add pooling layer for downsampling if required
    if pool:
        layer_name = (module_name + '_pool') if (module_name is not None) else None
        pool_output = MaxPooling2D(pool_size=pool_size, name=layer_name)(conv_output)
        return conv_output, pool_output
    return conv_output


def decompression_module_2D(decompression_input,
                            compression_input,
                            num_output_features,
                            kernel_size=(3, 3),
                            num_convs=2,
                            upsample_factor=(2, 2),
                            activation='relu',
                            module_name=None,
                            dropout_rate=None,
                            dropout_type='standard',
                            norm_type='batchnorm',
                            conv_kwargs=None,
                            norm_kwargs=None):
    '''Create an upsampling decompression module

    A decompression module sits on the right hand path of the model and increases
    the size of the input image by upsampling followed by merger with an output
    from the compression path of the same size

    Parameters:
    -----------
    decompression_input : keras layer
        The input layer that comes from the previous decompression module
    compression_input : keras layer
        The input layer that comes from the a layer in the compression path. Must have twice the height and width of
        the decompression_input layer
    num_output_features : int
        The number of output features (dimension of the features channel at the module's output)
    kernel_size : tuple
        Shape (height, width) of the convolution kernel in each conv layer
        (default 3x3)
    num_convs : int
        number of convolutional layers before the upample (default: 2)
    upsample_factor : tuple
        Upsampling factors (height, width) to use
        (default 2x2)
    activation : string or function
        The activation function used in each conv layer. May be any activation function
        recognized by Keras
    dropout_rate: None or float
        Dropout rate to apply after each activation. The dropout rate represents the fraction of units to drop at
        training time.
    dropout_type: str
        Type of dropout. Choose one of 'standard', 'gaussian' or 'alpha' to apply the Dropout, Gaussian or
        AlphaDropout Keras layers respectively. Note that to create a self-normalizing network (with the
        'selu' activation), the 'alpha' dropout option should be used.
    norm_type : str
        Type of normalization to apply before every activation layer, choose from '['none', 'batchnorm',
        'groupnorm', 'instancenorm'] (default: 'batchnorm')
    conv_kwargs : dict
        Allows passing arbitrary kwargs into the convolutional layers of the module (Keras Conv2D layers)
    norm_kwargs : dict
        Allows passing arbitrary kwargs into the normalization layers, if any, (whose type is determined by the
        'norm_type' parameter,
    module_name : str
        Name used as the basis for individual layer names within the module

    Returns:
    --------
    output
        keras layer after the two convolution operations. Width and height are the same as the compression_input,
        and the size of the channel dimension is num_output_features

    '''

    if conv_kwargs is None:
        conv_kwargs = {}
    if norm_kwargs is None:
        norm_kwargs = {}

    if K.image_data_format() == "channels_first":
        c_ax = 1
    elif K.image_data_format() == "channels_last":
        c_ax = 3

    # Upsampling and skip connection
    layer_name = (module_name + '_upsample') if (module_name is not None) else None
    upsampled = UpSampling2D(size=upsample_factor, name=layer_name)(decompression_input)
    layer_name = (module_name + '_concat') if (module_name is not None) else None
    output = concatenate([upsampled, compression_input], axis=c_ax, name=layer_name)

    # Convolutional layers
    for i in range(num_convs):
        # Convolutional
        layer_name = (module_name + '_conv_' + str(i)) if (module_name is not None) else None
        output = Conv2D(num_output_features, kernel_size, padding='same',
                        name=layer_name, **conv_kwargs)(output)

        # Batch norm, if required
        layer_name = (module_name + '_norm_' + str(i)) if (module_name is not None) else None
        output = apply_normalization(output, axis=c_ax, name=layer_name, norm_type=norm_type,
                                     params=norm_kwargs)

        # Activation
        layer_name = (module_name + '_activation_' + str(i)) if (module_name is not None) else None
        output = Activation(activation, name=layer_name)(output)
        if dropout_rate is not None:
            layer_name = (module_name + '_dropout_' + str(i)) if (module_name is not None) else None
            output = apply_dropout(output, dropout_rate, dropout_type=dropout_type, name=layer_name)

    return output
