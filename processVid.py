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
import ffmpeg

import find_speaker
import time
import json

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
    command = [
    ffmpeg_path, 
    '-i', video_path, 
    '-i', audio_path, 
    '-c:v', 'copy',  # Copy video codec
    '-c:a', 'aac',   # Encode audio to AAC
    '-y',            # Overwrite output file if it exists
    output_path
]
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
    while len(ms)<3:
        ms="0"+ms

    
    return hour+":"+minute+":"+second+"."+ms

def cut_audio(input_path, output_path, startF, endF):
    file_path=input_path
    start_time_ms = startF*40
    end_time_ms=endF*40
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Convert times from milliseconds to the AudioSegment format
    start_time = start_time_ms
    end_time = end_time_ms
    
    # Slice the audio
    sliced_audio = audio[start_time:end_time]
    
    # Export the sliced audio to a new file
    sliced_audio.export(output_path, format="wav")

'''
def cut_audio(input_path, output_path, startF, endF):
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    startC = time_str(startF/25.0)
    endC = time_str(endF/25.0)
    command = [
        ffmpeg_path,
        '-i', input_path,
        '-ss', startC,
        '-to', endC,
        '-c:a', 'aac',
        output_path
    ]
'''
    # Clean up temporary files
    
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
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    
    command = [
    ffmpeg_path,
    '-ss', str(startF/25.0),  # Start time for frame-accurate seeking
    '-i', video_path,       # Input file
    '-to', str(endF/25.0),    # End time for frame-accurate seeking
            
    '-g', '1',               # Forces keyframes on every frame for maximum accuracy
    '-c:v', 'libx264',       # Video codec, can use others like 'libx265' for HEVC
    '-c:a', 'copy',           # Audio codec, can use others like 'libmp3lame' for MP3
   
    '-movflags', '+faststart', # Optimize for streaming
    '-y',                    # Overwrite output file without asking
    output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stderr)

    
    '''
    start_time = startF/25.0
    end_time = endF/25.0
    mp4box_path = "/opt/homebrew/bin/MP4Box"# Path to the MP4Box executable if not in PATH

    inter_path = video_path.replace(".mp4", "_inter.mp4")
    
    # Construct the MP4Box command
    command = [
        mp4box_path,
        '-splitx', f'{start_time}:{end_time}',
        video_path,
        '-out', inter_path
    ]


    command1 = [
        ffmpeg_path,
        '-i', inter_path,       # Input file
        '-c:v', 'libx264',      # Video codec: H.264
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        '-c:a', 'aac',          # Audio codec (optional)
        '-movflags', '+faststart',  # Optimize for streaming
        '-y',                   # Overwrite the output file
        output_path
    ]

    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(f"Video successfully cut and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(e.stderr.decode())  # Print detailed error message from MP4Box

    
    try:
        result = subprocess.run(command1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(f"Video successfully converted and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(e.stderr.decode())  # Print the detailed error message from ffmpeg
    '''


    

def cut_video1(input_file, start_time, end_time, output_file):
    # Load the video
    video = VideoFileClip(input_file)

    # Cut the video between start_time and end_time
    cut = video.subclip(start_time, end_time)

    # Write the result to a file
    cut.write_videofile(output_file, codec="libx264", audio_codec="aac")

    # Close the video clip
    cut.close()


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
    min_size = 200
    try:
        faces = DeepFace.represent(img_path=image_path, model_name="ArcFace", detector_backend = 'retinaface')
    except:
        return []
    ret = []
    for i in range(len(faces)):
        face = faces[i]
        confidence = face['face_confidence']
        embedding = face['embedding']
        facial_area = face['facial_area']
        if(confidence>=0.8 and facial_area['w']>=200 and facial_area['h']>=200):
            ret.append([embedding, facial_area])
    return ret



def distance(old, new):
    return math.sqrt((old[0]-new[0])**2 + (old[1]-new[1])**2)

def compare_face(embedding1, embedding2):
    #the print statements is for testing
    cosine_similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cosine_similarity>=0.6

def cut_speaker_in(video_files):
    #this is a modification, it accept an entire array of video files
    model_name = "ArcFace"

    while i<len(video_files):
        video_path = video_files[i]
        cap = cv.VideoCapture(video_path)
        img_path = video_path+"image.jpg"

        #declare status variables for finding speaker
        speaker_face = find_speaker.find_speaker(video_path) #this should in theory return an embedding
        if(speaker_face == "no"):
            os.remove(video_path)
            continue
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
        i+=1
        print("finished cutting")

def draw_box(video_path, output_video_path, boxes_list, frame_rate=25):
    print("drawing")
    cap = cv.VideoCapture(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, 25, (frame_width, frame_height))
    for i in range(len(boxes_list)):
        ret, frame = cap.read()
        if not ret:
            print(f"break, {i}")
            print(cap.isOpened())
            break
        location = boxes_list[i]
        cv.rectangle(frame, (location[0], location[1]), (location[0]+location[2], location[1]+location[3]), (0, 0, 255))
        out.write(frame)
    out.release()
    cap.release()


def get_video_info(input_path):
    ffprobe_path = "/opt/homebrew/bin/ffprobe"
    command = [
        ffprobe_path, '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'json',
        input_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = json.loads(result.stdout.decode('utf-8'))['frames']
    keyframe_intervals = []
    last_keyframe = None
    frame_type = ""
    
    for i, frame in enumerate(frames):
        frame_type+=frame['pict_type']
    
    # Return the average keyframe interval
    print("frame rate and info")
    print(frame_type)

    command = [
        ffprobe_path, '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,bit_rate,codec_name',
        '-of', 'json',
        input_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    video_info = json.loads(result.stdout.decode('utf-8'))['streams'][0]
    print(video_info)


def check_video_integrity(video_path):
    #this is for testing, consider deleting later
    command = ['/opt/homebrew/bin/ffprobe', '-v', 'error', '-show_entries',
               'packet=pts_time,dts_time',
               '-of', 'csv=p=0', video_path]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.stderr:
        print("Errors detected in the video:")
        print(result.stderr)
    else:
        print("No obvious errors detected in the video.")
 

def track_and_cut1(video_path, old_folder, new_folder, box=False):
    print("starting")
    print(video_path)
    cap = cv.VideoCapture(video_path)
    get_video_info(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    audio_path = video_path.replace(".mp4", "_audio.aac")
    extract_audio(video_path, audio_path)
    frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    tracking = False
    seg = 0
    start_frame=0
    time1 = time.perf_counter()
    print("hi")
    locations = []
    print("hi")
    for i in range(frame_num):
        
        ret, frame = cap.read()
        if not ret:
            break
        img_path = video_path+".image.jpg"
        cv.imwrite(img_path, frame)
        faces = extract(img_path)
        os.remove(img_path)
        if(tracking==False) and len(faces)==1:
            tracking=True
            speaker=faces[0][0]
            location = faces[0][1]
            
            x=location['x']
            y=location['y']
            w=location['w']
            h=location['h']
            locations.append([x,y,w,h])
            start_frame=i
            
        elif(tracking==True) and len(faces)==1 and (find_speaker.compare_face(faces[0][0], speaker)>=0.7) and (i<(frame_num-1)):
            location = faces[0][1]
            x=location['x']
            y=location['y']
            w=location['w']
            h=location['h']
            locations.append([x,y,w,h])
            
        elif(tracking==True) and (i-start_frame>=25):
            print("cutting")
            print(str(i))
            print(str(start_frame))
            print(time_str(i/25.0))
            print(time_str(start_frame/25.0))
            current_frame = i

            temp_path = video_path.replace(".mp4", f"_{seg}temp.mp4").replace(old_folder, new_folder)
            seg_path = temp_path = video_path.replace(".mp4", f"_seg{seg}.mp4").replace(old_folder, new_folder)
                
            cut_video(video_path, temp_path, start_frame, current_frame)
            get_video_info(temp_path)
            time.sleep(1)
            scan = cv.VideoCapture(temp_path)
            out = cv.VideoWriter(seg_path, fourcc, 25, (frame_width, frame_height))
            
            if(box):
                
                
                #print(ffmpeg.probe(temp_path, v='error', show_streams=True))
                
                print(f"locations: {len(locations)}")
                #temp_path = os.path.abspath(temp_path)
                #draw_box(temp_path, seg_path, locations)

                for a in range(len(locations)):
                    #file exist and is readable, presumably permission is granted
                    ret, frame = scan.read()
                    
                    if not ret:
                        print(f"cut, {a}")
                        break
                    else:
                        print(f"read, {a}")
                    
                    location = locations[a]
                    cv.rectangle(frame, (location[0], location[1]), (location[0]+location[2], location[1]+location[3]), (0, 0, 255))
                    out.write(frame)     
                out.release()
                scan.release()
                temp_audio_path = video_path.replace(".mp4", f"_audio{seg}.mp4").replace(old_folder, new_folder)
                extract_audio(temp_path, temp_audio_path)
                combine_audio(seg_path, temp_audio_path, seg_path.replace("seg", "fin"))
                locations = []
                seg+=1
                tracking=False  
            else:
                seg_path = temp_path = video_path.replace(".mp4", f"_seg{seg}.mp4").replace(old_folder, new_folder)
                cut_video(video_path, seg_path, start_frame, i)
                
            
       
        elif(tracking==True):
            
            
            tracking==False
            
            locations=[]
            print("stopped tracking")
        else:
            print("nothing")

        if (i+1)%25==0 or (i+1)==frame_num:
            time2 = time.perf_counter()
            print(f"processed frame: {i}, time used: {time2-time1}")
            time1=time.perf_counter()



def track_and_cut(video_path, old_folder, new_folder):
    #this method is for when there is red boxes
    print("starting")
    print(video_path)
    cap = cv.VideoCapture(video_path)
    get_video_info(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    audio_path = video_path.replace(".mp4", "_audio.aac")
    extract_audio(video_path, audio_path)
    frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    tracking = False
    seg = 0
    start_frame=0
    time1 = time.perf_counter()
    print("hi")
    locations = []
    print("hi")
    segs = []
    for i in range(frame_num):
        
        ret, frame = cap.read()
        if not ret:
            break
        img_path = video_path+".image.jpg"
        cv.imwrite(img_path, frame)
        faces = extract(img_path)
        os.remove(img_path)
        
        if(tracking==False) and len(faces)==1:
            print(f"started {i}")
            tracking=True
            speaker=faces[0][0]
            location = faces[0][1]
            
            x=location['x']
            y=location['y']
            w=location['w']
            h=location['h']
            locations.append([x,y,w,h])
            start_frame=i
            
        elif(tracking==True) and len(faces)==1 and (find_speaker.compare_face(faces[0][0], speaker)>=0.7) and (i<(frame_num-1)):
            location = faces[0][1]
            x=location['x']
            y=location['y']
            w=location['w']
            h=location['h']
            locations.append([x,y,w,h])
            
        elif(tracking==True) and (i-start_frame>=25):
            print(f"cutting{i}")
            print(str(i))
            print(str(start_frame))
            print(time_str(i/25.0))
            print(time_str(start_frame/25.0))
            current_frame = i

            temp_path = video_path.replace(".mp4", f"_{seg}temp.mp4").replace(old_folder, new_folder)
                
            cut_video(video_path, temp_path, start_frame, i)    
            #print(ffmpeg.probe(temp_path, v='error', show_streams=True))
                
            print(f"locations: {len(locations)}")
            #temp_path = os.path.abspath(temp_path)
            
            segs.append([temp_path, locations])
            locations = []
            seg+=1
            tracking=False
        elif(tracking==True):     
            tracking=False   
            locations=[]
            print(f"track abandoned {i}")
        else:
            print("nothing")

        if (i+1)%25==0 or (i+1)==frame_num:
            time2 = time.perf_counter()
            print(f"processed frame: {i}, time used: {time2-time1}")
            time1=time.perf_counter()
    #at this point we start drawing
    for i in range(len(segs)):
        temp_path = segs[i][0]
        locations = segs[i][1]
        seg_path = temp_path.replace("temp", "seg")
        draw_box(temp_path, seg_path, locations)
        audio_path = temp_path.replace(".mp4", "_audio.mp4")
        extract_audio(temp_path, audio_path)
        combine_audio(seg_path, audio_path, seg_path.replace("seg", "fin"))
        cut_video(seg_path.replace("seg", "fin"), seg_path, 0, len(locations))
        os.remove(seg_path.replace("seg", "fin"))
        os.remove(temp_path)
        os.remove(audio_path)
        
            

def track_face(video_files):
    a=20
    while a<100:
        print("starting: "+str(a))
        video_path = video_files[a]
        time1 = time.perf_counter()
        #searching for the speaker face
        speaker_embedding = find_speaker.find_speaker(video_path)
        time2 = time.perf_counter()
        print("time taken searching: "+str(time2-time1))
        if speaker_embedding == "no":
            print("terminated")
            a+=1
            continue
        #looping through the video
        time1 = time.perf_counter()
        cap = cv.VideoCapture(video_path)
        
        # Get original video properties
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv.CAP_PROP_FOURCC))
        # Define the codec and create a temporary VideoWriter object
        output_path = video_path + '.track.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or use *'XVID' for .avi
        out = cv.VideoWriter(output_path, fourcc, 25, (width, height))
        
        #mark speaker in each frame, and then output
        frame_num = cap.get(cv.CAP_PROP_FRAME_COUNT)
        frame_count = 1
        time2 = time.perf_counter()
        print("time taken preparing loop: "+str(time2-time1))
        time1 = time.perf_counter()
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break
            
            img_path = video_path + ".temp.jpg"
            cv.imwrite(img_path, frame)
            faces = extract(img_path)
            speaker_i = find_speaker.compare_faces(speaker_embedding, faces)
            
            
            
            for i in range(len(faces)):
                location = faces[i][1]
                if(i==speaker_i):
                    cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (0, 0, 255), 2)
                else:
                    cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (255, 0, 0), 2)
            out.write(frame)
            
            if(frame_count%25 == 0 or frame_count == frame_num):
                time2 = time.perf_counter()
                print("processed frame: "+str(frame_count)+"/"+str(frame_num))
                print("time: "+str(time2-time1))
                time1=time.perf_counter()
            
            frame_count+=1
            os.remove(img_path)

        a+=1
        #release all
        out.release()
        cap.release()

        print("")

def match_audio(video_path):
    audio_path = video_path+".audio.aac"
    extract_audio(video_path, audio_path)
    video = AudioSegment.from_file(video_path, "mp4")
    audio = AudioSegment.from_file(audio_path, "aac")
    if abs(len(video)-len(audio))<=0.05:
        return true
    else:
        return false
    
    


# main code starts here

#define locations in the following 2 lines of code
#fill in your own directories
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
print("FFmpeg path:", os.getenv("IMAGEIO_FFMPEG_EXE"))
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.config as config
config.change_settings({"FFMPEG_BINARY": "/opt/homebrew/bin/ffmpeg"})



os.chdir("/Users/clarencewu/Downloads")
folder = "avspeech_part_c2"

directory = os.getcwd()
path = os.path.join(directory, folder)
print(f"Current directory: {directory}")
print(f"Folder path: {path}")
video_path = os.path.join(path, "-0yLDC7r3SI_109.066667_116.966667.mp4")
video_files = get_video_files(path)
from moviepy.editor import VideoFileClip
track_and_cut(video_path, "avspeech_part_c2", "avspeech_part_c2_processed")

'''

os.chdir("/Users/clarencewu/Downloads")
folder = "avspeech_part_c2"

directory = os.getcwd()
path = os.path.join(directory, folder)

if os.path.isdir(path):
    print(f"Selected folder: {path}")
    video_files = get_video_files(path)
    if video_files:
        print("Video files found.")
        for video in video_files:
            print(f"Processing video file: {video}")
            change_frame_rate(video, target_fps=25)
    else:
        print("No video files found in the selected folder.")
else:
    print(f"Folder '{folder}' does not exist in the current directory.")
'''

