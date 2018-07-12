# Do one time data preparation steps:
# - Resize images to 300x300.
# - Run through Inceptionv3 > 2048 N feature vector.
#
# CLEANUP(laser): Have this as a callable maybe_prepare_image_data(path) method
# that only processes the data if the file doesn't already exist

import os
import json
import numpy as np
import tensorflow as tf

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
    prepare_text_data()
