{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing import sequence\n",
    "from keras.models import Sequential, Model, load_model, model_from_yaml\n",
    "from keras.layers import Dense, Dropout, Activation\n",
    "from keras.layers import Embedding\n",
    "from keras.layers import Conv1D, GlobalMaxPooling1D\n",
    "from keras.datasets import imdb\n",
    "import tensorflow as tf\n",
    "from keras import backend as K\n",
    "\n",
    "tf.logging.set_verbosity(tf.logging.ERROR)\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from skater.core.local_interpretation.dnni.deep_interpreter import DeepInterpreter\n",
    "from skater.core.visualizer.text_relevance_visualizer import build_explainer, show_in_notebook\n",
    "from skater.util.dataops import convert_dataframe_to_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a TensorFlow session and register it with Keras. It will use this session to initialize all the variables\n",
    "sess = tf.Session()\n",
    "K.set_session(sess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# set parameters:\n",
    "max_features = 5000\n",
    "maxlen = 80\n",
    "batch_size = 32\n",
    "embedding_dims = 50\n",
    "filters = 250\n",
    "kernel_size = 3\n",
    "hidden_dims = 250\n",
    "epochs = 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load the Dataset\n",
    "#### IMDB dataset: \n",
    "##### 1. http://ai.stanford.edu/~amaas//data/sentiment/\n",
    "##### 2. http://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf ( Section 4.1 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading data...\n",
      "25000 train sequences\n",
      "25000 test sequences\n"
     ]
    }
   ],
   "source": [
    "# The Dataset contains 50,000 reviews(Train:25,000 and Test:25,000)\n",
    "# More info about the dataset: https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification\n",
    "\n",
    "print('Loading data...')\n",
    "(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)\n",
    "print(len(x_train), 'train sequences')\n",
    "print(len(x_test), 'test sequences')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<START> shown in australia as <UNK> this incredibly bad movie is so bad that you become <UNK> and have to watch it to the end just to see if it could get any worse and it does the storyline is so predictable it seems written by a high school <UNK> class the sets are pathetic but <UNK> better than the <UNK> and the acting is wooden br br the <UNK> <UNK> seems to have been stolen from the props <UNK> of <UNK> <UNK> there didn't seem to be a single original idea in the whole movie br br i found this movie to be so bad that i laughed most of the way through br br <UNK> <UNK> should hang his head in shame he obviously needed the money\n",
      "\n",
      "Length: 129\n"
     ]
    }
   ],
   "source": [
    "# https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset\n",
    "# Reading raw text\n",
    "INDEX_FROM = 3\n",
    "word_to_id = imdb.get_word_index()\n",
    "\n",
    "def get_raw_txt(word_id_dict, input_data):\n",
    "    word_id_dict = {k:(v+INDEX_FROM) for k,v in word_id_dict.items()}\n",
    "    word_id_dict[\"<PAD>\"] = 0\n",
    "    word_id_dict[\"<START>\"] = 1\n",
    "    word_id_dict[\"<UNK>\"] = 2\n",
    "    id_to_word = {value:key for key,value in word_id_dict.items()}\n",
    "    return ' '.join(id_to_word[_id] for _id in input_data)\n",
    "\n",
    "r_t = get_raw_txt(word_to_id, x_train[20])\n",
    "print(r_t + \"\\n\")\n",
    "print(\"Length: {}\".format(len(r_t.split(' '))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pad sequences (samples x time)\n",
      "x_train shape: (25000, 80)\n",
      "x_test shape: (25000, 80)\n"
     ]
    }
   ],
   "source": [
    "print('Pad sequences (samples x time)')\n",
    "x_train = sequence.pad_sequences(x_train, maxlen=maxlen)\n",
    "x_test = sequence.pad_sequences(x_test, maxlen=maxlen)\n",
    "print('x_train shape:', x_train.shape)\n",
    "print('x_test shape:', x_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<UNK> class the sets are pathetic but <UNK> better than the <UNK> and the acting is wooden br br the <UNK> <UNK> seems to have been stolen from the props <UNK> of <UNK> <UNK> there didn't seem to be a single original idea in the whole movie br br i found this movie to be so bad that i laughed most of the way through br br <UNK> <UNK> should hang his head in shame he obviously needed the money\n",
      "\n",
      "Length: 80\n"
     ]
    }
   ],
   "source": [
    "# Raw text post selecting the top most frequently occurring words\n",
    "r_t_r = get_raw_txt(word_to_id, x_train[20])\n",
    "print(r_t_r + \"\\n\")\n",
    "print(\"Length: {}\".format(len(r_t_r.split(' '))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Build model...\n"
     ]
    }
   ],
   "source": [
    "# Reference: https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py\n",
    "print('Build model...')\n",
    "model = Sequential()\n",
    "\n",
    "# we start off with an efficient embedding layer which maps\n",
    "# our vocab indices into embedding_dims dimensions\n",
    "model.add(Embedding(max_features,\n",
    "                    embedding_dims,\n",
    "                    input_length=maxlen))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "# we add a Convolution1D, which will learn filters\n",
    "# word group filters of size filter_length:\n",
    "model.add(Conv1D(filters,\n",
    "                 kernel_size,\n",
    "                 padding='valid',\n",
    "                 activation='relu',\n",
    "                 strides=1))\n",
    "\n",
    "# we use max pooling:\n",
    "model.add(GlobalMaxPooling1D())\n",
    "\n",
    "# We add a vanilla hidden layer:\n",
    "model.add(Dense(hidden_dims))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "# We project onto a single unit output layer, and squash it with a sigmoid:\n",
    "model.add(Dense(1))\n",
    "model.add(Activation('sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "model.compile(loss='binary_crossentropy',\n",
    "              optimizer='adam',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 25000 samples, validate on 25000 samples\n",
      "Epoch 1/2\n",
      "25000/25000 [==============================] - 20s - loss: 0.4717 - acc: 0.7606 - val_loss: 0.3596 - val_acc: 0.8415\n",
      "Epoch 2/2\n",
      "25000/25000 [==============================] - 20s - loss: 0.3024 - acc: 0.8710 - val_loss: 0.3510 - val_acc: 0.8465\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7fe1977f6978>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,\n",
    "          epochs=epochs,\n",
    "          validation_data=(x_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "24864/25000 [============================>.] - ETA: 0sTest score: 0.35099207803726196\n",
      "Test accuracy: 0.84652\n"
     ]
    }
   ],
   "source": [
    "score, acc = model.evaluate(x_test, y_test,\n",
    "                            batch_size=batch_size)\n",
    "print('Test score:', score)\n",
    "print('Test accuracy:', acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<keras.layers.embeddings.Embedding at 0x7f9cce0cdcc0>,\n",
       " <keras.layers.core.Dropout at 0x7f9cce39afd0>,\n",
       " <keras.layers.convolutional.Conv1D at 0x7f9cce063dd8>,\n",
       " <keras.layers.pooling.GlobalMaxPooling1D at 0x7f9cce396198>,\n",
       " <keras.layers.core.Dense at 0x7f9cce396780>,\n",
       " <keras.layers.core.Dropout at 0x7f9cce056dd8>,\n",
       " <keras.layers.core.Activation at 0x7f9cce003550>,\n",
       " <keras.layers.core.Dense at 0x7f9cce396ba8>,\n",
       " <keras.layers.core.Activation at 0x7f9cce003f28>]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.layers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Persist the model for future use"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Save model to disk\n"
     ]
    }
   ],
   "source": [
    "# Save and persist the trained keras model in YAML format\n",
    "model_yaml = model.to_yaml()\n",
    "with open(\"model_cnn_imdb_{}.yaml\".format(epochs), \"w\") as yaml_file:\n",
    "    yaml_file.write(model_yaml)\n",
    "# serialize weights to HDF5\n",
    "model.save_weights(\"model_cnn_imdb_{}.h5\".format(epochs))\n",
    "print(\"Save model to disk\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load the saved model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loaded model from disk\n",
      "24960/25000 [============================>.] - ETA: 0sTest score: 0.35099207803726196\n",
      "Test accuracy: 0.84652\n"
     ]
    }
   ],
   "source": [
    "# load the model\n",
    "K.set_learning_phase(0)\n",
    "yaml_file = open('model_cnn_imdb_{}.yaml'.format(epochs), 'r')\n",
    "loaded_model_yaml = yaml_file.read()\n",
    "yaml_file.close()\n",
    "loaded_model = model_from_yaml(loaded_model_yaml)\n",
    "# load weights into new model\n",
    "loaded_model.load_weights('model_cnn_imdb_{}.h5'.format(epochs))\n",
    "print(\"Loaded model from disk\")\n",
    "\n",
    "\n",
    "# Validate model performance with the reload of persisted model\n",
    "loaded_model.compile(loss='binary_crossentropy',\n",
    "              optimizer='adam',\n",
    "              metrics=['accuracy'])\n",
    "score, acc = loaded_model.evaluate(x_test, y_test,\n",
    "                            batch_size=batch_size)\n",
    "print('Test score:', score)\n",
    "print('Test accuracy:', acc)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Lets ask Skater to help us in interpreting the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "learning_phase 0\n",
      "Load model from disk\n",
      "1/1 [==============================] - 0s\n",
      "Predicted class : [[1]]\n",
      "Ground Truth: [1]\n"
     ]
    }
   ],
   "source": [
    "index = 1\n",
    "K.set_learning_phase(0)\n",
    "with DeepInterpreter(session=K.get_session()) as di:\n",
    "    print(\"learning_phase {}\".format(K.learning_phase()))\n",
    "    yaml_file = open('model_cnn_imdb_{}.yaml'.format(epochs), 'r')\n",
    "    loaded_model_yaml = yaml_file.read()\n",
    "    yaml_file.close()\n",
    "    \n",
    "    loaded_model = model_from_yaml(loaded_model_yaml)\n",
    "    # load weights into new model\n",
    "    loaded_model.load_weights('model_cnn_imdb_{}.h5'.format(epochs))\n",
    "    print(\"Load model from disk\")    \n",
    "    \n",
    "    # Input data\n",
    "    xs = np.array([x_test[index]])\n",
    "    ys = np.array([y_test[index]])\n",
    "\n",
    "    print('Predicted class : {}'.format(loaded_model.predict_classes(np.array([x_test[index]]))))\n",
    "    print('Ground Truth: {}'.format(ys))\n",
    "    \n",
    "    embedding_tensor = loaded_model.layers[0].output\n",
    "    input_tensor = loaded_model.layers[0].input\n",
    "    \n",
    "    embedding_out = di.session.run(embedding_tensor, {input_tensor: xs});\n",
    "    # Using Integrated Gradient for computing feature relevance\n",
    "    relevance_scores = di.explain('integ_grad', loaded_model.layers[-2].output * ys, \n",
    "                                  loaded_model.layers[1].input, embedding_out);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "as he <UNK> the one liners out i also like the scenes with <UNK> at the beginning find her very sexy when she's wearing all that <UNK> <UNK> i can't be the only one surely i personally think bride of <UNK> is a fantastic film total entertainment from start to finish great humour horror in equal measure at only <UNK> minutes long it never becomes boring or dull a personal favourite of mine watch it as soon as you can\n"
     ]
    }
   ],
   "source": [
    "# Retrieve the text\n",
    "r_t = get_raw_txt(word_to_id, x_test[index])\n",
    "print(r_t)\n",
    "words_ = r_t.split(' ')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    80.000000\n",
       "mean      0.000578\n",
       "std       0.008350\n",
       "min      -0.041212\n",
       "25%      -0.000527\n",
       "50%       0.000000\n",
       "75%       0.001007\n",
       "max       0.032800\n",
       "dtype: float64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# building a dataframe with columns 'features' and 'relevance scores'\n",
    "# Since, the relevance score is compute over the embedding vector, we aggregate it by computing 'mean'\n",
    "# over the embedding to get scalar coefficient for the features\n",
    "relevance_scores_df = pd.DataFrame(relevance_scores[0]).mean(axis=1)\n",
    "relevance_scores_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>relevance_scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>80.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.000578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.008350</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-0.041212</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-0.000527</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.001007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>0.032800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       relevance_scores\n",
       "count         80.000000\n",
       "mean           0.000578\n",
       "std            0.008350\n",
       "min           -0.041212\n",
       "25%           -0.000527\n",
       "50%            0.000000\n",
       "75%            0.001007\n",
       "max            0.032800"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# merging the dataframe columnwise\n",
    "words_df = pd.DataFrame({'features': words_})\n",
    "scores_df = pd.DataFrame({'relevance_scores': relevance_scores_df.tolist()})\n",
    "words_scores_df = words_df.join(scores_df)\n",
    "words_scores_df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualize the results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "words_scores_dict = convert_dataframe_to_dict('features', 'relevance_scores', words_scores_df)\n",
    "build_explainer(r_t, words_scores_dict, highlight_oov=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<body><div style=background-color:#F5F5F5; white-space: pre-wrap; font-size: 12pt; font-family: Verdana;\"><body><div style=background-color:#F5F5F5; white-space: pre-wrap; font-size: 12pt; font-family: Verdana;\"><span style=\"background-color: rgba(255, 242, 236, 0.5)\">as</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\">he</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\"><UNK></span> <span style=\"background-color: rgba(245, 250, 254, 0.5)\">the</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">one</span> <span style=\"background-color: rgba(255, 242, 235, 0.5)\">liners</span> <span style=\"background-color: rgba(242, 248, 253, 0.5)\">out</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">i</span> <span style=\"background-color: rgba(253, 210, 191, 0.5)\">also</span> <span style=\"background-color: rgba(255, 240, 233, 0.5)\">like</span> <span style=\"background-color: rgba(245, 250, 254, 0.5)\">the</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">scenes</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\">with</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\"><UNK></span> <span style=\"background-color: rgba(245, 249, 254, 0.5)\">at</span> <span style=\"background-color: rgba(245, 250, 254, 0.5)\">the</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\">beginning</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">find</span> <span style=\"background-color: rgba(255, 244, 239, 0.5)\">her</span> <span style=\"background-color: rgba(254, 216, 199, 0.5)\">very</span> <span style=\"background-color: rgba(255, 244, 238, 0.5)\">sexy</span> <span style=\"background-color: rgba(255, 244, 239, 0.5)\">when</span> <span style=\"background-color: rgba(255, 244, 239, 0.5)\">she's</span> <span style=\"background-color: rgba(240, 246, 253, 0.5)\">wearing</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">all</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">that</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\"><UNK></span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\"><UNK></span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">i</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">can't</span> <span style=\"background-color: rgba(255, 244, 239, 0.5)\">be</span> <span style=\"background-color: rgba(245, 250, 254, 0.5)\">the</span> <span style=\"background-color: rgba(217, 232, 245, 0.5)\">only</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">one</span> <span style=\"background-color: rgba(246, 250, 255, 0.5)\">surely</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">i</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\">personally</span> <span style=\"background-color: rgba(246, 250, 255, 0.5)\">think</span> <span style=\"background-color: rgba(223, 236, 247, 0.5)\">bride</span> <span style=\"background-color: rgba(255, 237, 229, 0.5)\">of</span> <span style=\"background-color: rgba(255, 237, 229, 0.5)\"><UNK></span> <span style=\"background-color: rgba(252, 167, 139, 0.5)\">is</span> <span style=\"background-color: rgba(245, 249, 254, 0.5)\">a</span> <span style=\"background-color: rgba(189, 21, 26, 0.5)\">fantastic</span> <span style=\"background-color: rgba(255, 235, 226, 0.5)\">film</span> <span style=\"background-color: rgba(224, 236, 248, 0.5)\">total</span> <span style=\"background-color: rgba(255, 242, 236, 0.5)\">entertainment</span> <span style=\"background-color: rgba(242, 248, 253, 0.5)\">from</span> <span style=\"background-color: rgba(245, 250, 254, 0.5)\">start</span> <span style=\"background-color: rgba(243, 248, 254, 0.5)\">to</span> <span style=\"background-color: rgba(255, 238, 230, 0.5)\">finish</span> <span style=\"background-color: rgba(252, 158, 128, 0.5)\">great</span> <span style=\"background-color: rgba(252, 152, 121, 0.5)\">humour</span> <span style=\"background-color: rgba(255, 238, 230, 0.5)\">horror</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\">in</span> <span style=\"background-color: rgba(247, 251, 255, 0.5)\">equal</span> <span style=\"background-color: rgba(244, 249, 254, 0.5)\">measure</span> <span style=\"background-color: rgba(245, 249, 254, 0.5)\">at</span> <span style=\"background-color: rgba(217, 232, 245, 0.5)\">only</span> <span style=\"background-color: rgba(217, 232, 245, 0.5)\"><UNK></span> <span style=\"background-color: rgba(205, 224, 241, 0.5)\">minutes</span> <span style=\"background-color: rgba(234, 243, 251, 0.5)\">long</span> <span style=\"background-color: rgba(254, 226, 213, 0.5)\">it</span> <span style=\"background-color: rgba(255, 238, 230, 0.5)\">never</span> <span style=\"background-color: rgba(232, 241, 250, 0.5)\">becomes</span> <span style=\"background-color: rgba(18, 93, 166, 0.5)\">boring</span> <span style=\"background-color: rgba(253, 206, 187, 0.5)\">or</span> <span style=\"background-color: rgba(8, 48, 107, 0.5)\">dull</span> <span style=\"background-color: rgba(245, 249, 254, 0.5)\">a</span> <span style=\"background-color: rgba(255, 242, 235, 0.5)\">personal</span> <span style=\"background-color: rgba(252, 187, 161, 0.5)\">favourite</span> <span style=\"background-color: rgba(255, 237, 229, 0.5)\">of</span> <span style=\"background-color: rgba(252, 142, 110, 0.5)\">mine</span> <span style=\"background-color: rgba(252, 188, 162, 0.5)\">watch</span> <span style=\"background-color: rgba(254, 226, 213, 0.5)\">it</span> <span style=\"background-color: rgba(255, 242, 236, 0.5)\">as</span> <span style=\"background-color: rgba(255, 244, 239, 0.5)\">soon</span> <span style=\"background-color: rgba(255, 242, 236, 0.5)\">as</span> <span style=\"background-color: rgba(255, 242, 236, 0.5)\">you</span> <span style=\"background-color: rgba(255, 245, 240, 0.5)\">can</span></div></body>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_in_notebook('./rendered.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
