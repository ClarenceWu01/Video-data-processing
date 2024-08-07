import tkinter as tk
from tkinter import filedialog
import os
import cv2 as cv
import subprocess
from pydub import AudioSegment
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm
from deepface import DeepFace
from scipy.spatial.distance import cosine

import find_speaker

'''
because my directory for ffmpeg is wrong, I manually set the directory in each function.
all functions that uses modified directories can be changed at the top of the function
'''

#the following segment is intended as a fix to incomplete read

import urllib3
from urllib3.exceptions import IncompleteRead

def patch_http_response_read(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IncompleteRead as e:
            return e.partial
    return inner

urllib3.HTTPResponse.read = patch_http_response_read(urllib3.HTTPResponse.read)


import logging
logging.basicConfig(level=logging.DEBUG)

#the patch ends here

def select_folder():
    # Create a root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Open the folder selection dialog
    folder_selected = filedialog.askdirectory()
    
    return folder_selected

def get_video_files(folder_path, extensions=['.mp4', '.avi', '.mkv']):
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def headless_files(folder_path, extensions=['.mp4', '.avi', '.mkv']):
    video_files=[]
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(file)
    return video_files

def extract_audio(video_path, audio_path):
    # Extract audio using ffmpeg
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    command = [ffmpeg_path, '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def combine_audio(video_path, audio_path, output_path):
    # Combine video and audio using ffmpeg
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    command = [ffmpeg_path, '-i', video_path, '-i', audio_path, '-c', 'copy', output_path, '-y']
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)



def change_frame_rate(video_path, target_fps=25):
    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get original video properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv.CAP_PROP_FOURCC))

    
    # Define the codec and create a temporary VideoWriter object
    temp_video_path = video_path + '.temp.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or use *'XVID' for .avi
    out = cv.VideoWriter(temp_video_path, fourcc, target_fps, (width, height))

    #extract audio for later
    audio_path = video_path + '.audio.aac'
    extract_audio(video_path, audio_path)
    
    # Read and write frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    #add audio back
    combine_audio(temp_video_path, audio_path, video_path)
    

    # Release everything when done
    cap.release()
    out.release()

    # delete files
    os.remove(temp_video_path)
    os.remove(audio_path)

def time_str(time):
    hour = str(math.floor(time/3600))
    minute = str(math.floor(time/60)%60)
    second = str(math.floor(time%60))
    ms = str(int(time%1 * 1000))
    while len(hour)<2:
        hour = "0"+hour
    
    while len(minute)<2:
        minute = "0"+minute
    
    while len(second)<2:
        second="0"+second
    while len(ms)<4:
        ms="0"+ms

    
    return hour+":"+minute+":"+second+"."+ms

def cut_video1(video_path, startT, endT):
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"

    # Extract audio
    audio_path = video_path + '.audio.aac'
    extract_audio(video_path, audio_path)

    # Cut audio
    startC = time_str(startT)
    endC = time_str(endT)
    cut_audio_path = video_path + '.cut_audio.aac'
    command = [
        ffmpeg_path,
        '-i', audio_path,
        '-ss', startC,
        '-to', endC,
        '-c', 'copy',
        cut_audio_path
    ]
    subprocess.run(command, check=True)

    # Cut video (and audio if you want to sync audio and video)
    temp_video_path = video_path + '.temp.mp4'
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-ss', startC,
        '-to', endC,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        temp_video_path
    ]
    subprocess.run(command, check=True)

    # Combine cut audio with cut video
    combine_audio(temp_video_path, cut_audio_path, video_path)

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(audio_path)
    os.remove(cut_audio_path)

'''
def cut_video(video_path, output_path, startF, endF):
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"

    # Extract audio
    audio_path = video_path + '.audio.aac'
    extract_audio(video_path, audio_path)

    # Cut audio
    startC = time_str(startF/25)
    endC = time_str(endF/25)
    cut_audio_path = video_path + '.cut_audio.aac'
    command = [
        ffmpeg_path,
        '-i', audio_path,
        '-ss', startC,
        '-to', endC,
        '-c', 'copy',
        cut_audio_path
    ]
    subprocess.run(command, check=True)

    # Cut video (and audio if you want to sync audio and video)
    temp_video_path = video_path + '.temp.mp4'
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-ss', startC,
        '-to', endC,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        temp_video_path
    ]
    subprocess.run(command, check=True)

    # Combine cut audio with cut video
    combine_audio(temp_video_path, cut_audio_path, output_path)

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(audio_path)
    os.remove(cut_audio_path)
'''
#rework version
def cut_video(video_path, output_path, startF, endF):
    startC = time_str(startF/25)
    endC = time_str(endF/25)
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    temp_video_path = video_path + '.temp.mp4'
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-ss', startC,
        '-to', endC,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(command, check=True)



def track_face1(video_path, folder_path, new_folder_path):
    # Load the video
    cap = cv.VideoCapture(os.path.join(folder_path, video_path))

    # Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    tracker = dlib.correlation_tracker()


    #variables
    tracking = False
    face_position = None
    frame_num = 0
    position = []
    speaker_known = False

    while not speaker_known:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        frame_num += 1
        gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces)==1:
            face_encodings = face_recognition.face_encodings(rgb_frame, [faces[0]])
            speaker_face = face_encodings[0]
            speaker_known = True

    #actul segmenting
    seg_num = 0
    frame_num=0
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        
        gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_encodings = face_recognition.face_encodings(rgb_frame, [faces[0]])
        if (len(faces)==1 and faces[0]==speaker_face and not tracking):
            tracking = True
            startF = frame_num
            endF = frame_num
        elif len(faces)==1 and faces[0]==speaker+face and tracking:
            endF += 1
        else:
            tracking = False
            if endF-startF>=10 :
                cut_video(os.path.join(folder_path, video_path), os.path.join(new_folder_path, video_path)+".seg"+str(seg_num)+".mp4", startF, endF)
                seg_num+=1

def extract(image_path):
    #temp_path = image_path + ".temp.jpg"
    #adjustable value: confidence
    faces = DeepFace.extract_faces(img_path=image_path, detector_backend = 'retinaface')
    ret = []
    for i in range(len(faces)):
        face = faces[i]
        confidence = face['confidence']
        temp_path = image_path + ".temp.jpg"
        face_array = face['face']
        cv.imwrite(temp_path, face_array)
        facial_area = face['facial_area']
        face_representation = DeepFace.represent(img_path=temp_path, model_name = "ArcFace", enforce_detection=False)
        #print(face_representation)
        #testing remove print(face_representation)
        if(confidence>=0.8 and facial_area['w']>=0 and facial_area['h']>=0):
            ret.append([face_representation[0]['embedding'], facial_area, 0]) #the final 0 is for the multi-frame determination system only, frames of appearence
        #the following line is strictly for debugging
        os.remove(temp_path)
    #the following line is strictly for debugging
    return ret


def distance(old, new):
    return math.sqrt((old[0]-new[0])**2 + (old[1]-new[1])**2)

def compare_face(embedding1, embedding2):
    #the print statements is for testing
    cosine_similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cosine_similarity>=0.6

def track_face2(video_path):
    model_name = "ArcFace"
    cap = cv.VideoCapture(video_path)
    img_path = video_path+"image.jpg"

    #declare status variables for finding speaker
    speaker_face = find_speaker.find_speaker(video_path) #this should in theory return an embedding

    #actul segmenting
    
    seg_num = 0
    startF = 0
    endF = 0
    endF
    tracking = False
    old_position = (0, 0)
    new_position = (0, 0)
    frame_num = 0
    print("started cutting")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imwrite(img_path, gray)
        faces = extract(img_path)
        
        if len(faces)==1 and compare_face(faces[0][0], speaker_face) and tracking == False:
            old_position = (faces[0][1]['x'], faces[0][1]['y'])
            new_position = (faces[0][1]['x'], faces[0][1]['y'])
            tracking=True
            startF = frame_num
            endF = frame_num
        elif len(faces)==1 and compare_face(faces[0][0], speaker_face) and distance(old_position, new_position)<=30 and tracking==True:
            old_position = new_position
            new_position = (faces[0][1]['x'], faces[0][1]['y'])
            endF +=1
        else:
            tracking = False
            if endF-startF>=10:
                cut_video(video_path, video_path+".seg"+str(seg_num)+".mp4", startF, endF)
                seg_num+=1
        frame_num+=1
    print("finished cutting") 
            

def track_face(video_path):
    # Load the video
    cap = cv.VideoCapture(video_path)

    # Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    tracker = dlib.correlation_tracker()

    
    #variables
    tracking = False
    face_position = None
    frame_num = 0
    position = []
    speaker_known = False

    while not speaker_known:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        frame_num += 1
        gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces)==1:
            face_encodings = face_recognition.face_encodings(rgb_frame, [faces[0]])
            speaker_face = face_encodings[0]
            speaker_known = True
    print("speaker known")
    #actul segmenting
    seg_num = 0
    frame_num=0
    
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        
        gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_encodings = face_recognition.face_encodings(rgb_frame, [faces[0]])
        if len(faces)==1 and faces[0]==speaker+face and not tracking:
            tracking = True
            startF = frame_num
            endF = frame_num
        elif len(faces)==1 and faces[0]==speaker+face and tracking:
            endF += 1
        elif i==frame_count--:
            tracking = False
            if endF-startF>=10 :
                cut_video(video_path, video_path+".seg"+str(seg_num)+".mp4", startF, endF)
                seg_num+=1
        else:
            tracking = False
            if endF-startF>=10 :
                cut_video(video_path, video_path+".seg"+str(seg_num)+".mp4", startF, endF)
                seg_num+=1
    cap.release()
    print("cutting finished")


def match_audio(video_path):
    audio_path = video_path+".audio.aac"
    extract_audio(video_path, audio_path)
    video = AudioSegment.from_file(video_path, "mp4")
    audio = AudioSegment.from_file(audio_path, "aac")
    if abs(len(video)-len(audio))<=0.05:
        return true
    else:
        return false
    
    
'''
def cut(video_path, startT, endT):
    #open file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    cap=cap[startT: endT]
    cap.export(video_path, format="mp4")
'''

# main code starts here

#define locations in the following 2 lines of code
#fill in your own directories
os.chdir("")
folder = ""

directory = os.getcwd()
path = os.path.join(directory, folder)
print(f"Current directory: {directory}")
print(f"Folder path: {path}")
video_files = get_video_files(path)

'''
if os.path.isdir(path):
    print(f"Selected folder: {path}")
    video_files = headless_files(path)
    new_folder = "avspeech_part_seg"
    if video_files:
        print("Video files found.")
        for video in video_files:
            print(f"Processing video file: {video}")
            #change_frame_rate(video, target_fps=25)
            track_face1(video, path, new_folder)
    else:
        print("No video files found in the selected folder.")
else:
    print(f"Folder '{folder}' does not exist in the current directory.")
'''
#the following codes will be a testing of track face 2
video = video_files[1]
track_face2(video)
