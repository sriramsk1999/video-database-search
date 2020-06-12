import cv2
import numpy as np
from sklearn.utils import shuffle
import os
import time
import json
from models import *
import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf

#to generate the caption from the meanpooled vector of a video segment
def evaluate_tensor(img_tensor_val):
    hidden = decoder.reset_state(batch_size=1)

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            result.pop()
            return ' '.join(result)

        dec_input = tf.expand_dims([predicted_id], 0)

    result.pop()
    return ' '.join(result)

# Download caption annotation files
annotation_folder = 'annotations/'
download_annotations = False
if download_annotations:
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                          extract = True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'
  os.remove(annotation_zip)
else:
  annotation_file = 'annotations/captions_train2017.json'

with open(annotation_file, 'r') as f:
    annotations = json.load(f)
t = time.time()

######################
#  Prepare tokenizer #
######################

# Store captions and image names in vectors
all_captions = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions = shuffle(all_captions, random_state=1)

#select only 250000
num_examples = 250000
train_captions = train_captions[:num_examples]

# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

print("Prepared tokenizer. Time taken : ",time.time() - t)
t = time.time()

######################
#    Load models     #
######################

embedding_dim = 256
vocab_size = 5001
units=512
max_length=52
features_shape = 2048
attention_features_shape = 64

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

encoder.load_weights('encoder/encoder')
decoder.load_weights('decoder/decoder')

print("Loaded models. Time taken : ",time.time() - t)
t = time.time()


######################
#  Prepare segments  #
######################

segmentList = []

PATH = '/home/ubuntu/dataset/'

category = ['sports/','news/','travel/','science/','people/','food/']

#1244 segments
for cat in category:
	path = PATH + cat
	for root,d_names,f_names in os.walk(path):
		for fname in f_names:
			if '_' in fname:
				segmentList.append(root+'/'+fname)

print("Prepared segments. Time taken : ",time.time() - t)
t = time.time()

######################
#  Generate Captions #
######################

segmentCaption = {}
video_count = 0
l = len(segmentList)

for seg in segmentList:
	count=0
	cap = cv2.VideoCapture(seg)
	featureList = []

	nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	while cap.isOpened():
		ret, frame = cap.read()
		print((float(count)/nframe)*100,"%",end='\r')
		if ret:
			#each frame is a numpy array
			frame = tf.image.resize(frame, (299, 299))
			frame = tf.keras.applications.inception_v3.preprocess_input(frame)

			temp_input = tf.expand_dims(frame, 0)
			img_tensor_val = image_features_extract_model(temp_input)
			img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

			featureList.append(img_tensor_val)
			count += 10 # i.e. get every tenth frame
			cap.set(1, count)
		else:
			cap.release()
			break

	print('                              \r100%')
	try:
		pool = tf.concat(featureList,0)
	except:
		print("ERROR ERROR ERROR")
		continue

	meanpool = tf.math.reduce_mean(pool,axis=0)
	meanpool = tf.expand_dims(meanpool,0)

	result = evaluate_tensor(meanpool)

	print(video_count,'/',l,'---',seg,'---',result)

	segmentCaption[seg] = result
	video_count+=1

with open('segmentCaption.json','w') as outfile:
	json.dump(segmentCaption,outfile)

print("Captioned videos. Time taken : ",time.time() - t)

