# USAGE
# python main.py train --style ./path/to/style/image.jpg(video.mp4)   \
#                      --dataset ./path/to/dataset \
#                      --weights ./path/to/weights  \
#                      --batch 2

# python main.py evaluate --content ./path/to/content/image.jpg   \
#                         --weights ./path/to/weights \
#                         --result ./path/to/save/results/image.jpg


import os
import argparse
from train import trainer
from evaluate import transfer
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Currently, memory growth needs to be the same across GPUs
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

CONTENT_WEIGHT = 6e0
STYLE_WEIGHT = 2e-3
TV_WEIGHT = 6e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 2

STYLE_IMAGE = './images/style/udnie.jpg'
CONTENT_IMAGE = './images/content/chicago.jpg'
DATASET_PATH = '../datasets/train2014'
RESULT_NAME = './images/results/Result.jpg'
WEIGHTS_PATH = None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fast Style Transfer')
    parser.add_argument('command',
                         metavar='<command>',
                         help="'train' or 'evaluate'")
    parser.add_argument('--debug', required=False, type=bool,
                         metavar=False,
                         help='Whether to print the loss',
                         default=False)
    parser.add_argument('--dataset', required=False,
                         metavar=DATASET_PATH,
                         default=DATASET_PATH)
    parser.add_argument('--style', required=False,
                         metavar=STYLE_IMAGE,
                         help='Style image to train the specific style',
                         default=STYLE_IMAGE) 
    parser.add_argument('--content', required=False,
                         metavar=CONTENT_IMAGE,
                         help='Content image/video to evaluate with',
                         default=CONTENT_IMAGE)  
    parser.add_argument('--weights', required=False,
                         metavar=WEIGHTS_PATH,
                         help='Checkpoints directory',
                         default=WEIGHTS_PATH)
    parser.add_argument('--result', required=False,
                         metavar=RESULT_NAME,
                         help='Path to the transfer results',
                         default=RESULT_NAME)
    parser.add_argument('--batch', required=False, type=int,
                         metavar=BATCH_SIZE,
                         help='Training batch size',
                         default=BATCH_SIZE)
    parser.add_argument('--max_dim', required=False, type=int,
                         metavar=None,
                         help='Resize the result image to desired size or remain as the original',
                         default=None)

    args = parser.parse_args()


    # Validate arguments
    if args.command == "train":
        assert os.path.exists(args.dataset), 'dataset path not found !'
        assert os.path.exists(args.style), 'style image not found !'
        assert args.batch > 0
        assert NUM_EPOCHS > 0
        assert CONTENT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert TV_WEIGHT >= 0
        assert LEARNING_RATE >= 0

        parameters = {
                'style_file' : args.style,
                'dataset_path' : args.dataset,
                'weights_path' : args.weights,
                'debug' : args.debug,
                'content_weight' : CONTENT_WEIGHT,
                'style_weight' : STYLE_WEIGHT,
                'tv_weight' : TV_WEIGHT,
                'learning_rate' : LEARNING_RATE,
                'batch_size' : args.batch,
                'epochs' : NUM_EPOCHS,
            }

        trainer(**parameters)


    elif args.command == "evaluate":
        assert args.content, 'content image/video not found !'
        assert args.weights, 'weights path not found !'

        parameters = {
                'content' : args.content,
                'weights' : args.weights,
                'max_dim' : args.max_dim,
                'result' : args.result,
            }

        transfer(**parameters)


    else:
        print('Example usage : python main.py evaluate --content ./path/to/content/image.jpg')
        
        
if __name__ == '__main__':
    main()
