#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:04:36 2020

@author: siddarthv
"""
import json
import os
import subprocess

def get_video(url, start, end, vid_id, category):
    
    folder = "{}/{}".format(category, vid_id)
    try:
        print("URL: {}\t Start: {} End: {}".format(url, start, end))
        output = subprocess.check_output(["youtube-dl", "-g", url]).decode("utf-8").split("\n")
        video_url = '"' + output[0] + '"'
        
        if not os.path.exists(folder):
            os.mkdir(folder)
                
        output_path = "{}/{}.mp4".format(folder, vid_id)
            
        cmd = ["ffmpeg", "-i", video_url, "-y", "-ss", start, "-to", end, output_path]
            
        with open(os.devnull, 'w') as FNULL:
            code = subprocess.check_call(" ".join(cmd), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
            if code == 0:
                print("Created", output_path)
            else:
                print("Error creating", output_path)
                    
    except Exception as e:
        print("Error found was:", e)

if __name__ == "__main__":
    with open("verified_video_data.json", "r") as f:
        verified = json.load(f)
    
    for category in verified:
        print()
        print("Category:", category)
        sampled_videos = verified[category]
        category_folder = category.split("/")[0]
        if not os.path.exists(category_folder):
            os.mkdir(category_folder)
        count = 0
        for video in sampled_videos:
            video_id = video["video_id"]
            url = video["url"]
            start = str(video["start time"])
            end = str(video["end time"])
            count += 1
            print("Video {} of 100:".format(count), end=" ")
            get_video(url, start, end, video_id, category_folder)