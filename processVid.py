import tkinter as tk
from tkinter import filedialog
import os
import cv2 as cv
import subprocess
from pydub import AudioSegment
import math

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
    if len(hour)<2:
        hour = "0"+hour
    if len(hour)<2:
        hour = "0"+hour
    minute = str(math.floor(time/60)%60)
    if len(minute)<2:
        minute = "0"+minute
    if len(minute)<2:
        minute = "0"+minute
    second = str(time%60)
    if len(second)<2:
        second = "0"+second
    if len(second)<2:
        second = "0"+second

    return hour+":"+minute+":"+second

def cut_video(video_path, startT, endT):
    ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
    #open file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
   

    #extract and audio
    startC = time_str(startT)
    endC = time_str(endT)
    print(startC)
    print(endC)
    
    audio_path = video_path + '.audio.aac'
    extract_audio(video_path, audio_path)
    command = [
        ffmpeg_path,
        '-i', audio_path,
        '-ss', startC,
        '-to', endC,
        '-c', 'copy',
        audio_path
    ]
    subprocess.run(command, check=True)

    #set up writer
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv.CAP_PROP_FOURCC))
    temp_video_path = video_path + '.temp.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or use *'XVID' for .avi
    out = cv.VideoWriter(temp_video_path, fourcc, 25, (width, height))

    #calculate frame position
    startF = startT*25
    endF = endT*25

    #cut video
    cap.set(cv.CAP_PROP_POS_FRAMES, startF)
    
    for i in range(startF, endF):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)


    #add audio back
    combine_audio(temp_video_path, audio_path, video_path)

    #release and delete
    cap.release()
    out.release()
    os.remove(temp_video_path)
    os.remove(audio_path)

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
os.chdir("/Users/clarencewu/Downloads")
folder = "avspeech_part"

directory = os.getcwd()
path = os.path.join(directory, folder)
print(f"Current directory: {directory}")
print(f"Folder path: {path}")


'''
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

video_files = get_video_files(path)
video = video_files[0]

cut_video1(video, 0, 1)
print(f"cutting video file: {video}")

print("complete")
