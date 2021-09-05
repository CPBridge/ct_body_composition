import argparse

from body_comp.train.slice_selection import train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a densenet regression model to predict the selection offset')
    parser.add_argument('data_dir', help='Location of the training data directory')
    parser.add_argument('model_output_dir', help='Location where trained models are to be stored')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of training epochs')
    parser.add_argument('--name', '-a', help='weights will be stored with this name')
    parser.add_argument('--gpus', '-g', type=int, default=1, help='number of gpus')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--threshold', '-t', type=float, default=10.0,
                        help='soft-threshold the distance with a sigmoid function with this scale parameter')
    parser.add_argument('--load_weights', '-w', help='load these weights and continue training')
    parser.add_argument('--initial_epoch', '-i', type=int, default=0, help='begin with this epoch number')
    parser.add_argument('--nb_layers_per_block', default=12, type=int,
                        help="number of layers per block in densenet or resnext")
    parser.add_argument('--nb_blocks', default=4, type=int, help="number of layers of blocks in densenet and resnext")
    parser.add_argument('--levels', '-L', nargs='+', help="Names of levels of interest as a space-separated list, e.g. L3 T5 T8 T10. Default: L3")
    parser.add_argument('--nb_initial_filters', default=16, type=int,
                        help="number of initial filters in densenet and resnext")
    parser.add_argument('--growth_rate', default=12, type=int, help="densenet growth rate (k) parameter")
    parser.add_argument('--compression_rate', default=0.5, type=float, help="densenet compression rate parameter")
    parser.add_argument('--initializer', '-I', default='glorot_uniform', help="initializer for weights in the network")
    parser.add_argument('--activation', '-A', default='relu', help="activation for units in the network")
    parser.add_argument('--omit_batch_norm', '-B', action='store_false', dest='batch_norm',
                        help="omit batch normalization")
    args = parser.parse_args()

    model = train(**vars(args))
