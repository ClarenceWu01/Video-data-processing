from deepface import DeepFace
import cv2 as cv
import numpy as np
import math
import os
from numpy import dot
from numpy.linalg import norm


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
        #don't change confidence, I think it is unnecessary. size caps change at your own discretion
        if(confidence>=0.8 and facial_area['w']>=0 and facial_area['h']>=0):
            ret.append([face_representation[0]['embedding'], facial_area, 0]) #the final 0 is for the multi-frame determination system only, frames of appearence
        #the following line is strictly for debugging
        os.remove(temp_path)
    #the following line is strictly for debugging
    return ret

def compare_face(embedding1, embedding2):
    #the print statements is for testing
    cosine_similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cosine_similarity>=0.6

def compare_face1(target, faces):
    #it is intended that the target would be the candidate being looked as. the 0.7 criteria is flexible
    max_match = 0.7
    max_index = -1
    for i in range(len(faces)):
        face=faces[i]
        cosine_similarity = dot(target[0], face[0]) / (norm(target[0]) * norm(face[0]))
        if cosine_similarity>max_match:
            max_match = cosine_similarity
            max_index = i
    return max_index

#this is a new method introduced for multi-frame determination
def add_diff_face(known, new_face):
    #this function is also to for multi-frame determination. the result would account for movement and appereance times.
    #the face format can be found in the extract method
    similar = False
    for face in known:
        if compare_face(face[0], new_face[0]):
            similar = True
    if not similar:
        known.append(new_face)

def check_face(candidates, faces):
    for i in range(len(candidates)):
        candidate = candidates[i]
        result = compare_face1(candidate, faces)
        if(result>-1):
            face = faces[result]
            candidate[1]=face[1]
            candidate[2]+=1
            faces.pop(result)
    # return candidates

def chose_speaker(candidates):
#this looks at the final scores of candidates and pick the highest
    max_votes=0
    max_index=0
    for i in range (len(candidates)):
        votes = candidates[i][2]
        if votes>max_votes:
            max_votes = votes
            max_index = i
    #for testing, it will return the max index given that the candidates index will evenly match up with the first frame index. 
    return max_index


def find_speaker(video_path):
    #this version is to be integrated with the rest of the script
    print("starting")
    print(video_path)
    cap = cv.VideoCapture(video_path)
    img_path = video_path+".image.jpg"


    ret, frame = cap.read()

    cv.imwrite(img_path, frame)
    #cv.imshow('input frame', frame)
    faces = extract(img_path)



    ret, frame = cap.read()
    cv.imwrite(img_path, frame)
    candidates = extract(img_path)
    print("candidate obtained")
    print("available candidates: "+str(len(candidates)))
    cap.release()
    os.remove(img_path)

    cap = cv.VideoCapture(video_path)
    print("filtering candidates")
    for i in range(10):
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
