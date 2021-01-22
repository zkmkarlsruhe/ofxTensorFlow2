from modules.utils import tensor_to_image, load_img, clip_0_1, resolve_video
from modules.forward import feed_forward
import tensorflow as tf

image_type = ('jpg', 'jpeg', 'png', 'bmp')


def transfer(content, weights, max_dim, result):

    if content[-3:] in image_type:

        # Build the feed-forward network and load the weights.
        network = feed_forward()
        network.load_weights("model_check").expect_partial()

        # Load content image.
        image = load_img(path_to_img=content, max_dim=max_dim, resize=False)
        print(image.shape)
        print('Transfering image...')
        # Geneerate the style imagee
        image = network(image)

        # Clip pixel values to 0-255
        image = clip_0_1(image)

        # Save the style image
        tensor_to_image(image).save(result)

    else:
        network = feed_forward()
        network.load_weights(weights)

        resolve_video(network, path_to_video=content, result=result)