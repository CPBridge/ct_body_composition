from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
import warnings


class MultiGPUModelCheckpoint(ModelCheckpoint):
    """Save a single copy of the weights for a model split over several GPUs after every epoch.
    Assumes that the multi-GPU model was created using the keras.utils.multi_gpu_model function.
    Otherwise, this is identical to the ModelCheckpoint callback.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def set_model(self, model):
        # Overwrite the self.model variable with the original model
        # In the multi-gpu model, the original model is located before the output layers
        self.model = model.layers[-len(model.outputs) - 1]
        if not isinstance(self.model, Model) and not isinstance(self.model, Sequential):
            warnings.warn('The layer before the outputs is not a keras model type. Are you sure the model was created'
                          'using multi_gpu_model?', RuntimeWarning)
