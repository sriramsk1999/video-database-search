#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:56:35 2020

@author: siddarthv
"""
import json
import subprocess

selected_categories = ["sports/actions", "people", "travel", "news/events/politics", "science/technology", "food/drink"]
sample_size = 100

all_categories = []
with open("category.txt", "r") as f:
    for line in f:
        all_categories.append(line.split()[0])
    
with open("videodatainfo_2017.json", "r") as f:
    d = json.load(f)
        
    print("Number of videos:", len(d["videos"]))
    print("Number of sentences:", len(d["sentences"]))
        
    print()
        
video_data = d["videos"]    
sentence_data = d["sentences"]
    
categorical_videos = {}
for video in video_data:
    category = all_categories[video["category"]]
    if category not in categorical_videos:
        categorical_videos[category] = []
    categorical_videos[category].append(video)

verified = {}
for category in selected_categories:
    verified[category] = []
    for video in categorical_videos[category]:
        url = video["url"]
        print(video["video_id"])
        try:
            output = subprocess.check_output(["youtube-dl", "-g", url, "--force-ipv4"])
            verified[category].append(video)
        except Exception as e:
            print(e)
        if len(verified[category]) == sample_size:
            break

for category in verified:
    print(category, len(verified[category]))
with open("verified_video_data.json", "w") as f:
    json.dump(verified, f)