import json
import tensorflow as tf
import numpy as np

# Common paths.
# TODO(laser): Convert these to arguments for the script.
datasets_path = 'D:/Datasets'
captions_path = datasets_path + '/annotations/captions_train2014.json'
image_features_path = 'image_features.npy'

# Config
# TODO(laser): Temporary small values here.
max_images_to_process = 256

# OPTIMIZE(laser): Data prep step should have everything written to disk in a
# format that tf.data.Dataset can stream. Investigate how Tensorflow does that
# as loading the whole thing into memory takes around 1-1.5 GB of RAM. Also,
# caption processing done here is naive and super slow.
def load_data(captions_path, image_features_path):
    '''Loads the COCO dataset. Expects a path to one of its json files and to
    a numpy array generated by prepare_image_data.py.'''
    captions = {}
    vocabulary = {}
    sentences_dict = {}
    word_count = 0
    image_count = 0

    # Load image feature vectors. The loaded numpy array should have image_id as
    # the first column followed by 2048 columns with feature data. ids are
    # sorted to match captions to images easier later on.
    image_features = np.load(image_features_path)
    ids = image_features[:, 0]
    image_features = image_features[:, 1 : 2049]

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
            image_count = image_count + 1
            if image_count >= max_images_to_process:
                break

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

    # Dataset sanity check. Log if any images are lacking captions. Load only
    # valid data into the dataset and ignore any incomplete pairs.
    idx_to_remove = []
    for idx in range(len(ids)):
        if not ids[idx] in sentences_dict:
            idx_to_remove.append(idx)

    if len(idx_to_remove) > 0:
        print('WARNING: %d images were found without captions, ignoring them.'
              % (len(idx_to_remove)))
        ids = np.delete(ids, idx_to_remove, axis=0)
        image_features = np.delete(image_features, idx_to_remove, axis=0)

    sentences = [sentences_dict[v] for v in sorted(sentences_dict.keys())]
    pad = len(max(sentences, key=len))
    sentence_array = np.array([i + [-1]*(pad-len(i)) for i in sentences])
    dataset = tf.data.Dataset.from_tensor_slices(
        {'image_id': ids,
         'img_features': image_features,
         'sentences': sentence_array})

    return dataset, vocabulary

# Build Model
dataset, vocabulary = load_data(captions_path, image_features_path)


