# VETE

Attempt at implementing the Visually Enhanced Text Embeddings model to train BoW embeddings.


## Dependencies

tensorflow-gpu, numpy, matplotlib, pillow, h5py. (and their dependencies, of course)

The COCO2014 train and val datasets, found here http://cocodataset.org/#download. Note, the paper claims COCO2014 has 540k images for the training dataset, download only provided 83k images.

Other datasets can be used in theory, but prepare_data.py assumes the formatting is the same as COCO2014. Changes are likely to be required for other datasets to work.

This prototype was not tested on cpu implementations of tensorflow, and according to the documentation, it should throw errors during a tf.gather layer that purposefully has out of bounds indices.


## Configuration

A `paths` objects are found in vete.py with inputs for the program to run. `prepare_data.py` can be run directly but a copy of `paths` object must be modified there as well at the bottom of the file.

The prototype does image data prep in about 1-2 hours on a GTX 1070, text data prep in a few minutes and the training step in 1-2 minutes. Image data and text data prep are cached and only run once. The bottleneck for image data prep is actually on CPU for reading and resizing images, InceptionV3 runs much faster than that for feature extraction.


## Challenges during development

The model is presented rather clear in the paper. Elements that were not specified:

 - Batch size used during training. The model requires several sentences to be trained at once but the batch size hyperparameter was not mentioned. Did not observe any significant difference between 32, 64, 1024 during testing. I haven't actually compared this properly, yet I believe the dataset is too small to infer anything about required batch size.
 - How sentence similarity is computed for the validation step. They test the embeddings against several datasets, such as SemEval. This has 5 levels of similarity attached to pairs of sentences, yet there's no clear description of how they used the word embeddings to evaluate the sentences against those datasets. I've only attempted to replicate the COCOTest described in the paper, which is a binary test.
 - If any hidden layers are present in the model. There's no mention of anything like that, but some of the individual steps could have hidden layers.

The last layer in this implementation of InceptionV3 is of variable length. Took the second to last layer which has (as stated in the paper) the correct size of 2048. Skimmed the InceptionV3 paper, but now I kinda want to double check that is the correct data. This was done early in the development and as I'm writing this I have doubts that was correct.

The COCO dataset had 10 images which resulted in no features detected by InceptionV3. Removed those from the dataset to avoid divisions by 0 in the cosine similarity.

Have ran into several issues during validation. I believe some bugs are still hidden in here and I do not believe the model actually computes useful embeddings due to it.

 - Assumed sentence embeddings were done the same as for training. Normalized sums of word embeddings.
 - Assumed cosine similarity was the correct measurement for similarity.
 - Pretty much all pairs of sentences obtained over 95% similarity for both random pairs and similar pairs.
 - Similar pairs obtained from the same dataset will share multiple words. This causes them to have a slightly higher similarity than the shuffled pairs. Even with random word embeddings.
 - Training seems to somewhat separate the two distributions, but also moves every pair closer to 100% similar. I do not trust the model is working correctly due to this fact.
 - The value used to separate similar and dissimilar sentences is arbitrary, somewhere close to the middle of the initial distribution with random word embeddings. I don't trust it.

I've also messed up the computation of both the cosine similarity and pearson coefficient, but those were easy bugs to spot.

This is actually the first proper model I've implemented. First time using Tensorflow. Had several issues as one might expect with a new API. Main pain point was feeding data:

 - Attempted to use tf.data.Dataset first, but that would load everything in memory, run much slower and fail to allocate the full dataset. Resorted to feed_dict, but has the issue of not being able to provide data for a (batch_size, None) shaped Placeholder, with variable lengths for axis 1.
