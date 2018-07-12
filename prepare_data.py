# Do one time data preparation steps:
# - Resize images to 300x300.
# - Run through Inceptionv3 > 2048 N feature vector.
# - Parse captions file to sort and keep only 1 caption per image.

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

def prepare_text_data(captions_path='D:\\Datasets\\annotations\\captions_train2014.json',
                      image_features_path='image_features.npy',
                      output_path='text_data.json'):

    if os.path.isfile(output_path):
        print('Text prep: Output file already exists, doing nothing.')
        return

    captions = {}
    vocabulary = {}
    sentences_dict = {}
    word_count = 0

    # Load image feature vectors. The loaded numpy array should have image_id as
    # the first column followed by 2048 columns with feature data. ids are
    # sorted to match captions to images easier later on.
    image_features = np.load(image_features_path)
    ids = image_features[:, 0]

    # Process captions. We only want to keep 1 caption per image and they are
    # unsorted in the COCO dataset. We also want to build the vocabulary here
    # and convert the captions into arrays of indexes in the vocabulary. Ignore
    # captions not in the image_features array.
    with open(captions_path) as f:
        image_metadata = json.load(f)

        for caption_entry in image_metadata['annotations']:
            if not caption_entry['image_id'] in ids or \
                    caption_entry['image_id'] in captions:
                continue

            captions[caption_entry['image_id']] = caption_entry

    print('Captions filtered.')

    for image_id, caption_entry in captions.items():
        if 'caption' not in caption_entry:
            continue

        words = caption_entry['caption'].split(' ')
        sentence = []
        for word in words:
            if not word in vocabulary:
                vocabulary[word] = word_count
                word_count = word_count + 1
            sentence.append(vocabulary[word])
        sentences_dict[image_id] = sentence

    print('Vocabulary and sentence dict ready.')

    ids = sorted(sentences_dict.keys())
    sentences = [sentences_dict[v] for v in ids]

    with open(output_path, 'w') as outfile:
        json.dump({'vocabulary': vocabulary, 'ids':ids, 'sentences':sentences}, outfile)


if __name__ == '__main__':
    prepare_image_data()
    prepare_text_data()
