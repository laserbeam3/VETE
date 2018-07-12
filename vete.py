import json
import tensorflow as tf
import numpy as np
from prepare_data import prepare_image_data, prepare_text_data, \
    prepare_validation_data
from matplotlib import pyplot

# Common paths.
# OPTIONAL(laser): Convert these to arguments for the script.
image_features_path = 'image_features.npy'
text_data_path = 'text_data.json'
validation_data_path = 'validation_data.json'

# Config
batch_size = 1024
max_images_to_process = 10000000
num_epochs = 20
img_feature_count = 2048
word_feature_count = 128

# OPTIMIZE(laser): For larger datasets, we don't want to keep everything needed
# for feed_dict in memory, or we want to not use feed_dict at all.
def load_data(image_features_path, text_data_path, validation_data_path):
    '''Loads the COCO dataset. Expects a path to one of its json files and to ,
    validation_data_patha
    numpy array generated by prepare_image_data.py.'''

    prepare_image_data(output_path=image_features_path)
    prepare_text_data(output_path=text_data_path)

    with open(text_data_path) as f:
        text_data = json.load(f)

    vocabulary = text_data['vocabulary']
    sentence_ids = frozenset(text_data['ids'][:max_images_to_process])
    sentences = text_data['sentences'][:max_images_to_process]

    prepare_validation_data(vocabulary, output_path=validation_data_path)
    with open(validation_data_path) as f:
        test_data = json.load(f)

    # Load image feature vectors. The loaded numpy array should have image_id as
    # the first column followed by 2048 columns with feature data. ids are
    # sorted to match captions to images easier later on.
    img_features_feed = np.load(image_features_path)
    image_ids         = img_features_feed[:, 0].tolist()
    img_features_feed = img_features_feed[:, 1 : 2049]

    # Dataset sanity check. Log if any images are lacking captions. Load only
    # valid data into the dataset and ignore any incomplete pairs.
    idx_to_remove = []
    sentence_idx_to_remove = []
    for idx in range(len(image_ids)):
        # INVESTIGATE(laser): It appears that some of the images have no
        # features detected and are in fact 0 length vectors. There's a chance
        # this is a bug in prepare_data.py. For now remove them from the dataset
        # as they cause divisions by 0
        if np.count_nonzero(img_features_feed[idx]) == 0:
            idx_to_remove.append(idx)
            # Dataset used here only had 10 full zero vectors, not worried about
            # performance.
            try:
                found = text_data['ids'].index(image_ids[idx])
                sentence_idx_to_remove.append(found)
            except:
                pass
        elif image_ids[idx] not in sentence_ids:
            idx_to_remove.append(idx)

    for idx in sorted(sentence_idx_to_remove, reverse=True):
        del sentences[idx]

    # FIXME(laser): This layer only works on GPU implementations of Tensorflow,
    # according to docs this crashes when running on CPU. Extra -1s in the array
    # cauze additional 0s to be added in an np.gather step where this data is
    # used. A normalized sum follows and those 0s have no impact.
    pad = len(max(sentences, key=len))
    sentences_feed = np.array([idx + [-1]*(pad-len(idx)) for idx in sentences])

    if len(idx_to_remove) > 0:
        print('WARNING: %d images were found without captions, ignoring them.'
              % (len(idx_to_remove)))
        image_ids = np.delete(image_ids, idx_to_remove, axis=0)
        img_features_feed = np.delete(img_features_feed, idx_to_remove, axis=0)


    img_features_ph  = tf.placeholder(tf.float32, shape=(batch_size,
                                                         img_feature_count, ))
    sentences_ph     = tf.placeholder(tf.int32, shape=(batch_size, None))

    placeholders = {'img_features': img_features_ph,
                    'sentences': sentences_ph}
    feeds = {'img_features': img_features_feed,
             'sentences': sentences_feed}

    print("Data loaded and filtered.")
    return placeholders, feeds, vocabulary, test_data


def sentence_embedding(word_embeddings, sentence, axis=1):
    E_bow_gather = tf.gather(word_embeddings, sentence)
    E_bow_sum    = tf.reduce_sum(E_bow_gather, axis=axis)
    return tf.nn.l2_normalize(E_bow_sum, axis=axis)


def cosine_similarity(a, b, axis=1):
    dot = tf.reduce_sum(tf.multiply(a, b), axis)
    norm_a = tf.norm(a, axis=axis)
    norm_b = tf.norm(b, axis=axis)
    return dot / (norm_a * norm_b)


def build_vete_model(placeholders, vocabulary_size):
    model = {'layers': {}, 'variables': {}, 'placeholders': placeholders}

    img_features_ph   = placeholders['img_features']
    sentences_ph      = placeholders['sentences']

    half_batch = int(batch_size/2)
    word_count = len(vocabulary)

    W_img = tf.Variable(tf.random_uniform([img_feature_count,
                                          word_feature_count]))
    word_embeddings = tf.Variable(tf.random_uniform([word_count,
                                                     word_feature_count]))

    # The paper shuffles the image-text pairs half way through the model.
    # There's really no need to do that so late, and we can keep all the
    # computation linear if we shuffle before we touch any of our training
    # variables.
    labels        = tf.constant([1.0] * half_batch + [-1.0] * half_batch)

    img_a, img_b  = tf.split(img_features_ph, num_or_size_splits=2, axis=0)
    img_shuffled  = tf.random_shuffle(img_b)
    img_merged    = tf.concat([img_a, img_shuffled], axis=0)
    img_embedding = tf.matmul(img_merged, W_img)

    sentence_emb  = sentence_embedding(word_embeddings, sentences_ph)

    # INVESTIGATE(laser): Why does
    # tf.contrib.metrics.streaming_pearson_correlation break model gradients?
    def pearson(a, b, axis=0):
        mean_a, var_a = tf.nn.moments(a, axis)
        mean_b, var_b = tf.nn.moments(b, axis)
        cov = 1.0 / (batch_size-1) * tf.reduce_sum((a - mean_a) * (b - mean_b))
        return cov / tf.sqrt(var_a * var_b)

    similarity = cosine_similarity(img_embedding, sentence_emb)
    loss = pearson(similarity, labels)
    optimizer = tf.train.AdamOptimizer().minimize(-loss)

    model['layers']['img_features']     = img_features_ph
    model['layers']['sentences']        = sentences_ph
    model['layers']['img_a']            = img_a
    model['layers']['img_b']            = img_b
    model['layers']['img_shuffled']     = img_shuffled
    model['layers']['img_merged']       = img_merged
    model['layers']['img_embedding']    = img_embedding
    model['layers']['sentence_emb']     = sentence_emb
    model['layers']['similarity']       = similarity

    model['variables']['word_embeddings'] = word_embeddings
    model['variables']['W_img']           = W_img

    model['optimizer'] = optimizer
    model['loss']      = loss

    return model


# CLEANUP(laser): I'm manually running evaluation here, instead of building a
# model that I can call .predict() on. revisit here once more familiar with
# Tensorflow and how it does things.
# OPTIMIZE(laser): No need to recreate the model every time
def evaluate_model(vete_model, test_data, session, epoch):
    words       = tf.identity(vete_model['variables']['word_embeddings'])
    sentence1   = tf.placeholder(tf.int32, shape=(None))
    sentence2   = tf.placeholder(tf.int32, shape=(None))
    emb1        = sentence_embedding(words, sentence1, axis=0)
    emb2        = sentence_embedding(words, sentence2, axis=0)
    sim         = cosine_similarity(emb2, emb1, axis=0)
    are_similar = tf.round(sim * 0.507)  # This is definitely wrong and won't generalize...

    s1 = []
    s2 = []

    correct = 0
    for pair in test_data:
        _, s, r = session.run([words, sim, are_similar],
                              feed_dict={sentence1: pair['captions'][0],
                                         sentence2: pair['captions'][1]})
        if r == pair['similarity']:
            correct += 1

        if pair['similarity'] == 1:
            s1.append(s)
        else:
            s2.append(s)

    bins = np.linspace(0.94, 1, 400)
    pyplot.clf()
    pyplot.hist(s1, bins, normed=1, facecolor='green', alpha=0.5)
    pyplot.hist(s2, bins, normed=1, facecolor='blue', alpha=0.5)
    pyplot.savefig("results_%d.png" % (epoch))

    return float(correct)/float(len(test_data))


if __name__ == '__main__':
    placeholders, feeds, vocabulary, test_data= load_data(image_features_path,
                                                          text_data_path,
                                                          validation_data_path)
    model = build_vete_model(placeholders, len(vocabulary))

    img_feed        = feeds['img_features']
    sentences_feed  = feeds['sentences']
    img_features_ph = model['placeholders']['img_features']
    sentences_ph    = model['placeholders']['sentences']
    img_count       = len(sentences_feed)

    with open(validation_data_path) as f:
        test_data = json.load(f)

    # Run session
    with tf.Session() as s:
        print("Run training for %d samples over %d epocs. Batch size: %d" %
              (img_count, num_epochs, batch_size))

        s.run(tf.global_variables_initializer())
        s.run(tf.local_variables_initializer())

        print("Untrained accuracy: ", evaluate_model(model, test_data, s, 0))

        permutation    = np.random.permutation(img_count)
        img_feed       = img_feed[permutation]
        sentences_feed = [sentences_feed[x] for x in permutation]

        for epoch in range(1, num_epochs+1):
            print("Epoch ", epoch)
            loss_sum   = 0.0
            loss_count = 0.0
            for a in range(0, img_count - batch_size + 1, batch_size):
                b = a + batch_size
                _, loss = s.run([model['optimizer'], model['loss']],
                                feed_dict={img_features_ph: img_feed[a:b],
                                           sentences_ph: sentences_feed[a:b]});
                loss_sum += loss
                loss_count += 1

            print("Pearson correlation: ", loss_sum / loss_count)
            print("Accuracy: ", evaluate_model(model, test_data, s, epoch))
