from deepface import DeepFace
import cv2 as cv
import numpy as np
import math
import os
from numpy import dot
from numpy.linalg import norm
import time


from scipy.spatial.distance import cosine
model_name = "ArcFace"


def get_video_files(folder_path, extensions=['.mp4', '.avi', '.mkv']):
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def extract(image_path):
    #temp_path = image_path + ".temp.jpg"
    #adjustable value: confidence
    #called by: find speaker
    min_size=200
    try:
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend = 'retinaface')
    except:
        return ["no", "no"]
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
        #don't change confidence, I think changing is unnecessary. size caps change at your own discretion
        if(confidence>=0.8 and facial_area['w']>=min_size and facial_area['h']>=min_size):
            ret.append([face_representation[0]['embedding'], facial_area, 0]) #the final 0 is for the multi-frame determination system only, frames of appearence
        os.remove(temp_path)
    
    return ret

def extract_test(image_path):

    try:
        faces = DeepFace.extract_faces(img_path=image_path, detector_backend = 'retinaface')
    except:
        return []
    ret = []
    for i in range(len(faces)):
        face = faces[i]
        confidence = face['confidence']
        temp_path = image_path + ".temp.jpg"
        face_array = face['face']
        cv.imwrite(temp_path, face_array)
        facial_area = face['facial_area']
        face_representation = DeepFace.represent(img_path=temp_path, model_name = "ArcFace", enforce_detection=False)
        if(confidence>=0.8):
            ret.append([face_representation[0]['embedding'], facial_area, facial_area['w'], facial_area['h']]) #the final 0 is for the multi-frame determination system only, frames of appearence
        os.remove(temp_path)
    return ret


    
def compare_face(embedding1, embedding2):
    #the print statements is for testing
    #called by: none
    cosine_similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cosine_similarity>=0.7

def compare_faces(target, faces):
    #it is intended that the target would be the embedding of the candidate being looked as. the 0.7 criteria is flexible
    #called by: processVid, check_face
    max_match = 0.7
    max_index = -1
    for i in range(len(faces)):
        face=faces[i]
        cosine_similarity = dot(target, face[0]) / (norm(target) * norm(face[0]))
        if cosine_similarity>max_match:
            max_match = cosine_similarity
            max_index = i
    return max_index

#this is a new method introduced for multi-frame determination
def add_diff_face(known, new_face):
    #this function is also to for multi-frame determination. the result would account for movement and appereance times.
    #the face format can be found in the extract method
    #called by: none
    similar = False
    for face in known:
        if compare_face(face[0], new_face[0]):
            similar = True
    if not similar:
        known.append(new_face)

def check_face(candidates, faces):
    #called by: find_speaker
    for i in range(len(candidates)):
        candidate = candidates[i]
        result = compare_faces(candidate[0], faces)
        if(result>-1):
            face = faces[result]
            candidate[1]=face[1]
            candidate[2]+=1
            faces.pop(result)
    # return candidates

def chose_speaker(candidates):
    #this looks at the final scores of candidates and pick the highest
    #called by: find_speaker
    max_votes=0
    max_index=0
    for i in range (len(candidates)):
        votes = candidates[i][2]
        if votes>max_votes:
            max_votes = votes
            max_index = i
    #for testing, it will return the max index given that the candidates index will evenly match up with the first frame index. 
    return max_index

#this is the rework version
#def find_speakear(video_path):
    
def find_speaker_single(video_path):
    #this version is to be integrated with the rest of the script
    print("starting")
    print(video_path)
    cap = cv.VideoCapture(video_path)
    img_path = video_path+".image.jpg"

    '''
    ret, frame = cap.read()

    cv.imwrite(img_path, frame)
    #cv.imshow('input frame', frame)
    faces = extract(img_path)
    '''


    ret, frame = cap.read()
    cv.imwrite(img_path, frame)
    candidates = extract(img_path)
    print("candidate obtained")
    print("available candidates: "+str(len(candidates)))
    cap.release()
    os.remove(img_path)

    
    if len(candidates)==1:
        return candidates[0][0]
    else:
        return "no"

    #the below segment is commented out to only keep 1 speaker

def find_speaker_test(video_path):
    print("starting")
    print(video_path)
    cap = cv.VideoCapture(video_path)
    img_path = video_path+".image.jpg"

    ret, frame = cap.read()
    cv.imwrite(img_path, frame)
    candidates = extract_test(img_path)
    
    
    os.remove(img_path)

    if len(candidates)>1:
        print("frame 0, multiple person")
        status = "multiple"
    elif len(candidates)==0:
        print("frame 0, no speaker")
        status = "no"
    elif candidates[0][2]<200 or candidates[0][3]<200:
        print("frame 0, too small")
        status = "small"
    else:
        status = "pass"
        candidate = candidates[0]

    
    frame_num = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    
    output_path = video_path + ".temp.mp4"

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, 25, (frame_width, frame_height))
    print("frame num: " + str(int(frame_num)))
    time1 = time.perf_counter()
    for i in range(1, int(frame_num)):
        ret, frame = cap.read()
        if not ret:
            break
        img_path = video_path+".temp.jpg"
        if(not status=="pass"):
            out.write(frame)
            continue
        

        cv.imwrite(img_path, frame)
        faces=extract_test(img_path) 
        if len(faces)>1:
            status = "multiple"
            out.write(frame)
            print("frame "+str(i)+" multiple faces")
        elif len(faces)==0 or not compare_face(faces[0][0], candidate[0]):
            status = "left"
            out.write(frame)
            print("frame "+str(i)+" speaker left")
        elif faces[0][2]<200 or faces[0][3]<200:
            status = "small"
            out.write(frame)
            print("frame "+str(i)+" face too small")
        else:
            location = faces[0][1]
            cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (0, 0, 255), 2)
            out.write(frame)
        os.remove(img_path)
       
        if((i+1)%25==0 or (i+1)==frame_num):
            time2=time.perf_counter()
            print("processed frames: "+str(i+1))
            print("time: "+str(time2-time1))

    os.remove(video_path)
    os.rename(output_path, video_path+"."+status+".mp4")
    cap.release()
    out.release()


os.chdir("/Users/clarencewu/Downloads")
folder = "avspeech_part_c1"

directory = os.getcwd()
path = os.path.join(directory, folder)
print(f"Current directory: {directory}")
print(f"Folder path: {path}")
video_files = get_video_files(path)
print(len(video_files))
for i in range(len(video_files)):
    print("running video "+str(i))
    video=video_files[i]
    find_speaker_test(video)
    print("")
    
        
    '''
    cap = cv.VideoCapture(video_path)
    print("filtering candidates")
    for i in range(10):
        if len(candidates) ==1:
            break
        ret, frame = cap.read()
        if not ret:
            break
        img_path = video_path+".image.jpg"
        cv.imwrite(img_path, frame)
        faces = extract(img_path)
        check_face(candidates, faces)
        os.remove(img_path)
        print("filtered frame: " + str(i))
    cap.release()


    speaker_i = chose_speaker(candidates)
    print("votes casted")

    return candidates[speaker_i][0]

    #identify which face belongs to the speaker. the largest face is assumed to be the speaker's
    #the above idea is scratched
    '''

    '''
    max_size=0
    max_index=-1
    for i in range(len(faces)):
        size = faces[i][1]['w']*faces[i][1]['h']
        if size>max_size:
            max_index=i
            max_size=size

    location = faces[max_index][1]

    for i in range(len(faces)):
        location = faces[i][1]
        if(not i==max_index):
            cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (255, 0, 0), 2)
        else:
            cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (0, 0, 255), 2)
    '''
    '''
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cv.imwrite(img_path, frame)
    faces = extract(img_path)
    #look for the face of the known speaker, and then
    for i in range(len(candidates)):
        location = candidates[i][1]
        if(not i==speaker_i):
            cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (255, 0, 0), 2)
            #print("not speaker")
            #print(candidates[i][3])
        else:
            cv.rectangle(frame, (location['x'], location['y']), (location['x']+location['w'], location['y']+location['h']), (0, 0, 255), 2)
            #print("speaker")
            #print(candidates[i][3])
    cap.release()
    os.remove(img_path)

    output_path = video_path + ".output.jpg"
    cv.imwrite(output_path, frame)
    #cv.imshow('Output Frame', frame)
    #os.remove(output_path)
    '''
