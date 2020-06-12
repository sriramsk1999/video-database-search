import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import numpy as np
import time
from glob import glob
from PIL import Image
import pickle
from models import *
from nltk.translate.meteor_score import single_meteor_score 

#generates a string from a list of id's
def make_sentence(target_val):
  target_val = target_val.numpy()
  sentence=[]
  for i in target_val:
    if(i==0):
      sentence.pop()
      sentence.pop(0)
      return ' '.join(sentence)
    sentence.append(tokenizer.index_word[i]) 
  sentence.pop()
  sentence.pop(0)
  return ' '.join(sentence)

#write losses to json
def write_loss(loss,val_loss):
  loss = loss.numpy()
  f = open("loss.json","r")
  json_ob = json.load(f)
  loss_list = [str(i) for i in json_ob['loss']]
  val_loss_list = [str(i) for i in json_ob['val_loss']]
  f.close()
  loss_list.append(str(loss))
  val_loss_list.append(str(val_loss))
  f = open("loss.json","w")
  dictionary = {"loss":loss_list,"val_loss":val_loss_list}
  json_ob = json.dumps(dictionary)
  f.write(json_ob)
  f.close()

#predicts caption for image
def evaluate(img_tensor_val):
    hidden = decoder.reset_state(batch_size=1)

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            result.pop()
            return result

        dec_input = tf.expand_dims([predicted_id], 0)
    result.pop()
    return result

# prepares image for input
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


PATH = 'train2017/'
annotation_file = 'annotations/captions_train2017.json'

# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + '%012d.jpg' % (image_id)
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together and set a random state.
# train_captions and img_name vector have length ~ 600000, with each caption mapped 
# to a corresponding iamge path. Note: Since there are more captions, there are
# duplicate entries in img_name_vector
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

#select only 250000
num_examples = 250000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]


#Prepare model
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Get unique images
encode_train = sorted(set(img_name_vector))

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)

#following code only has to be run once, then files are stored on disk
features_stored = True
if(not(features_stored)): #features aren't stored on disk
	for img, path in image_dataset:
	  batch_features = image_features_extract_model(img)
	  batch_features = tf.reshape(batch_features,
	                              (batch_features.shape[0], -1, batch_features.shape[3]))
	  for bf, p in zip(batch_features, path):
	    path_of_feature = p.numpy().decode("utf-8")
	    np.save(path_of_feature, bf.numpy())


# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = max(len(t) for t in train_seqs)


# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.1,
                                                                    random_state=0)


# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 2000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
num_val_steps = 500
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)


val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
# Use map to load the numpy files in parallel
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#models are declared in models.py
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

checkpoint_path = "./checkpoints"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])*10
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

EPOCHS = 200

for epoch in range(start_epoch, EPOCHS):
    start = time.time() 
    total_loss = 0
    avg_meteor_score = 0

    #training
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
        	print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    print ('Epoch {} Loss {:.6f}'.format(epoch,
                                         total_loss/num_steps))    
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    #validation
    val_start = time.time()
    for (batch, (img_tensor_val, target_val)) in enumerate(val_dataset):
        prediction = ' '.join(evaluate(img_tensor_val))
        target = make_sentence(target_val)
        meteor_score = round(single_meteor_score(prediction, target),6)
        avg_meteor_score += meteor_score 
        if(batch==500): #only validate for first 500
          break
    print ('Epoch {} Avg Meteor Score {:.6f}'.format(epoch,
                                         avg_meteor_score/num_val_steps))
    print ('Time taken for validation {} sec\n'.format(time.time() - val_start))

    write_loss(total_loss/num_steps,avg_meteor_score/num_val_steps)
    if (epoch%10 == 0):
      ckpt_manager.save()
      encoder.save_weights('encoder/encoder',save_format='tf')
      decoder.save_weights('decoder/decoder',save_format='tf')

print('Complete!')
