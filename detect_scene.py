#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:23:47 2020

@author: siddarthv
"""

from scipy.spatial import distance as dist
import cv2
import sys
import subprocess
import time
import os


def split_video(filename, times):
    extension = filename.split(".")[-1]
    base_filename = ".".join(filename.split(".")[:-1])
    for index, (start, length) in enumerate(times):
        split_cmd = ["ffmpeg", "-i", filename, "-y", "-ss", str(start)]
        if length >= 0.0:
            split_cmd.extend(["-t", str(length)])
        split_filename = "{}{}.{}".format(base_filename, index+1, extension)
        split_cmd.append(split_filename)
        
        
        
        with open(os.devnull, 'w') as FNULL:
            code = subprocess.check_call(split_cmd, stdout=FNULL, stderr=subprocess.STDOUT)
            if code == 0:
                print("Created", split_filename)
            else:
                print("Error creating", split_filename)

def detect_frames(video_path, selected_method):
    cap = cv2.VideoCapture(video_path) # Open video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    
    prev_hist, prev_time, prev_diff = None, None, None
    METHODS = { # Set of opencv methods
    	"Correlation": cv2.HISTCMP_CORREL,
    	"ChiSquared": cv2.HISTCMP_CHISQR,
    	"Intersection": cv2.HISTCMP_INTERSECT,
    	"Hellinger": cv2.HISTCMP_BHATTACHARYYA, 
    	"Euclidean": dist.euclidean, # Set of scipy methods
    	"Manhattan": dist.cityblock,
    	"Chebyshev": dist.chebyshev}
    method = METHODS["Euclidean"]
    
    time_frame = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], # Generates a 3D histogram
    		[0, 256, 0, 256, 0, 256]) 
        hist = cv2.normalize(hist, hist).flatten() # Normalizes and converts to 1D vector
        frame_time, frame = cap.get(cv2.CAP_PROP_POS_MSEC), int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if prev_hist is not None:
            diff = method(prev_hist, hist)
            if prev_diff is not None and (diff - prev_diff) > 0.3:
                time_frame.append([*prev_time])
            prev_diff = diff
        prev_hist = hist
        prev_time = (round(frame_time)/1000, frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Creating an array of start times along with length
    split_times = []
    prev_frame_time = 0.00
    for frame_time, frame in time_frame:
        length = round(frame_time - prev_frame_time, 2)
        split_times.append((prev_frame_time, length))
        prev_frame_time = frame_time
    split_times.append((prev_frame_time, -1.0))
    return split_times

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    print("USAGE: python3 detect_scene.py <video_file_path>")
    sys.exit()

start = time.time()

split_times = detect_frames(video_path, "Euclidean")
split_video(video_path, split_times)

end = time.time()
print("Execution time: {} seconds".format(end - start))