# Do one time data preparation steps:
# - Resize images to 300x300.
# - Run through Inceptionv3 > 2048 N feature vector.

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications import inception_v3

def concat_np_list(list_of_np_arrays):
    shape = list(list_of_np_arrays[0].shape)
    shape[:0] = [len(list_of_np_arrays)]
    return np.concatenate(list_of_np_arrays).reshape(shape)

def prepare_image_data(images_root='D:\\Datasets\\train2014',
                       captions_path='D:\\Datasets\\annotations\\captions_train2014.json',
                       output_path='image_features.npy'):

    if os.path.isfile(output_path):
        print('Image prep: Output file already exists, doing nothing.')
        return

    # 'avg_pool' is the final layer in InceptionV3 before 'predictions'. That is the
    # data used by VETE.
    # NOTE(laser): This will download InceptionV3 and depends on pillow and h5py
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('avg_pool').output)

    with open(captions_path) as f:
        image_metadata = json.load(f)
    path = os.path.join(images_root, image_metadata['images'][0]['file_name'])

    chunk_size = 512
    image_ids = []
    result_list = []

    # OPTIMIZE(laser): In theory, image loading here is super slow. We're doing at
    # least 2-3 times the number of copies we need. In practice, this only needs to
    # run once over night.
    chunk_idx = 0
    for start in range(0, len(image_metadata['images']), chunk_size):
        image_count = 0
        image_list = []
        for image_entry in image_metadata['images'][start:start + chunk_size]:
            path = os.path.join(images_root, image_entry['file_name'])

            # NOTE(laser): Paper mentions rescaling to 300x300 but default arguments
            # in InceptionV3 docs say 299x299. Using that instead.
            img = image.load_img(path, target_size=(299, 299))
            x = image.img_to_array(img)
            image_ids.append(image_entry['id'])
            image_list.append(x)
            image_count += 1
            if image_count == chunk_size:
                chunk_idx += 1
                print('Loaded %s images (chunk %d)' %
                      (chunk_size * chunk_idx, chunk_idx - 1))
                break

        data = concat_np_list(image_list)
        data = inception_v3.preprocess_input(data)
        result = model.predict(data)
        result_list.append(result)
        print('Processed %s images (chunk %d)' %
              (chunk_size * chunk_idx, chunk_idx - 1))

    final_result = np.concatenate(result_list)
    final_result = np.insert(final_result, 0, np.array(image_ids), axis=1)
    final_result = np.sort(final_result, axis=0)
    np.save(output_path, final_result)

if __name__ == '__main__':
    prepare_image_data()
