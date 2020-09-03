from tensorflow.keras import backend as K


# Constant to avoid dividing by zero
smooth = 1.0


def dice_coef_multiclass_2D(y_true, y_pred):
    """A keras implementation of the multiclass Dice coefficient

    Adds up Dice coefficients for each non-background class individually. Note there is a small value added to the
    denominator to avoid division by zero, so this value should not be reported as the true Dice coefficient
    (the difference will be negligible for large arrays).

    Parameters:
    -----------
    y_true : keras layer
        The true classes
    y_pred : keras layer
        The keras layer that computes the classification softmax values

    Returns:
    --------
    keras layer
       Multiclass Dice coefficient output calculated across every pixel in the batch

    """

    if K.image_data_format() == "channels_first":
        b_ax, h_ax, w_ax, c_ax = 0, 2, 3, 1
    elif K.image_data_format() == "channels_last":
        b_ax, h_ax, w_ax, c_ax = 0, 1, 2, 3

    # Flatten predictions, preserving the class dimension
    y_pred_f = K.batch_flatten(K.permute_dimensions(y_pred, (c_ax, b_ax, h_ax, w_ax)))

    # Number of output classes
    num_classes = y_pred.shape[c_ax]

    # Create a one hot coded array of the same shape for the ground truth
    y_true = K.one_hot(K.cast(K.squeeze(y_true, axis=c_ax), dtype='uint8'), num_classes)
    true_one_hot = K.permute_dimensions(y_true, (3, 0, 1, 2))
    true_one_hot = K.batch_flatten(true_one_hot)

    # Find dice coeffcient for each class individually
    # Ignore class 0, assumed to be background
    class_losses = []
    for c in range(1, num_classes):
        this_class_intersection = K.sum(true_one_hot[c, :] * y_pred_f[c, :])
        this_class_loss = (2. * this_class_intersection + smooth) /\
                          (K.sum(true_one_hot[c, :]) + K.sum(y_pred_f[c, :]) + smooth)
        class_losses.append(this_class_loss)

    # Total loss is sum of class losses
    total_loss = class_losses[0]
    for cl in class_losses[1:]:
        total_loss += cl

    return total_loss


def dice_coef_multiclass_loss_2D(y_true, y_pred):
    """A keras implementation of the multiclass Dice loss

    Exactly the same as dice_coef_multiclass but returns -1 times the combined dice coefficient, making this function
    suitable for use as a loss function to be minimized within Keras.

    Parameters:
    -----------
    y_true : keras layer
        The true classes
    y_pred : keras layer
        The keras layer that computes the classification softmax values

    Returns:
    --------
    keras layer
       Multiclass Dice loss output calculated across every pixel in the batch

    """

    return -dice_coef_multiclass_2D(y_true, y_pred)
