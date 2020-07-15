import os
from nltk.translate.meteor_score import single_meteor_score
from collections import OrderedDict
import speech_recognition as sr
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess

app = Flask(__name__)

segmentCaption = OrderedDict(json.load(open('final.json','r')))

@app.route('/')
@app.route('/index.html')
def index():
	return render_template('index.html')

@app.route('/speech2text', methods=['POST'])
def speech2text():
	r = sr.Recognizer()

	audio = request.get_data()

	with open('tempo.ogg', mode='bx') as f:
		f.write(audio)

	# os.system('opusdec --force-wav tempo.ogg - | sox - tempo.wav')
	with open(os.devnull, 'w') as FNULL:
		cmd = ["ffmpeg", "-i", "tempo.ogg", "tempo.wav"]
		code = subprocess.check_call(" ".join(cmd), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

	try:
		audiofile = sr.AudioFile('tempo.wav')
		with audiofile as source:
			audio = r.record(source)

	except Exception as e:
		print("Error:", e)
		os.system("rm tempo.ogg")
		return "Could not process query"

	try:
		text = r.recognize_google(audio)
	except:
		text = "Could not process query"

	os.system('rm tempo.ogg tempo.wav')

	return text

@app.route('/proc_query')
def proc_query():
	query = request.args['query']
	score_list = []
	for i in segmentCaption:
		score_list.append(round(single_meteor_score(segmentCaption[i],query),6))

	top3 = sorted(score_list,reverse=True)[:3]

	t1 = score_list.index(top3[0])
	score_list.pop(t1)
	t2 = score_list.index(top3[1])
	score_list.pop(t2)
	t3 = score_list.index(top3[2])

	t1,t2,t3 = list(segmentCaption.items())[t1], list(segmentCaption.items())[t2], list(segmentCaption.items())[t3]
	#t1,t2,t3 hold top 3 results as a tuple
	#t1[0] - path to video
	#t1[1] - caption of video

	return jsonify(first=t1,
				   second=t2,
				   third=t3)


@app.route('/home/ubuntu/dataset/<path:videoname>')
def get_videos(videoname):
	return send_from_directory('/home/ubuntu/dataset',videoname)

app.run(host='0.0.0.0',debug=True)
