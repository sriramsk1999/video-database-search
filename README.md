## Video Database Search

A tool to search a raw video database through natural language queries.


#### Image Captioning

download_coco.py - downloads the images and the captions (MS-COCO)

img_caption.py - trains an encoder-decoder image captioning model using the data in annotations/ and train2017/

models.py - Contains definitions for the encoder and decoder models 

gen_caption.py - evaluate image captioner with a test image

----------

annotations - captions for images

encoder, decoder - checkpointed values for the two models

train2017 - images



#### Video Captioning

verify_dataset.py -> Verified the links provided in the dataset and saved the new verified links in a new json file

download_dataset.py -> Downloaded dataset from these verified links

detect_scene.py - splits a video into a series of chunks

detect_scene_multiple.py - segment the downloaded videos into chunks

caption_videos.py - uses the chunks in specified lcoations to generate a json file of captions for each chunk

----------


impapp/impapp.py - flask web app

impapp/templates/index.html - webpage for flask app

final.json - the final json storing {'videopath':'caption'}
 
