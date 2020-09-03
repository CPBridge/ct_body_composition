import argparse

from body_comp.train.segmentation import train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a u-net model on multiple classes')
    parser.add_argument('data_dir', help='Directory in which the segmentation training data arrays are stored')
    parser.add_argument('model_output_dir', help='Directory in which trained models should be saved')
    parser.add_argument('--apply_window_function', '-p', action='store_true',
                        help='apply a CT windowing function to images before training')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size','-b', type=int, default=16, help='batch size')
    parser.add_argument('--load_weights','-w', help='load weights in this file to initialise model')
    parser.add_argument('--name','-a', help='trained model will be stored in a directory with this name')
    parser.add_argument('--gpus','-g', type=int, default=1, help='number of gpus')
    parser.add_argument('--learning_rate','-l', type=float, default=0.1, help='learning rate')
    parser.add_argument('--decay_half_time','-d', type=float, default=20,
                        help='number of epochs until learning rate should halve')
    parser.add_argument('--compression_channels','-C', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512],
                        help='list of (space separated) feature dimensions up the compression path')
    parser.add_argument('--decompression_channels','-M', type=int, nargs='+', default=[256, 128, 64, 32, 16],
                        help='list of (space separated) feature dimensions up the decompression path')
    parser.add_argument('--activation','-A', default='relu', help='activation function to use')
    parser.add_argument('--num_convs','-N', default=1, type=int, help='activation function to use')

    args = parser.parse_args()

    model = train(**vars(args))
