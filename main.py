import cv2
import time
import numpy as np
import mediapipe as mp
import pyautogui as pag

# Intializing the Constants
MOTION_UP = "Up"
MOTION_DOWN = "Down"
MOTION_LEFT = "Left"
MOTION_RIGHT = "Right"
NO_MOTION = "_"
 


def perfrom(action,x1,y1,x2,y2):
    """
    Function to perform the action as per the motion

    @param binary: action that has performed
    @param x1,y1: (optional) the previous coordinates of the hand
    @param x2,y2: (optional) the current coordinates of the hand
    """
    if(action == MOTION_LEFT):
        pag.press('up',presses=3)
    elif(action==MOTION_RIGHT):
        pag.press('down',presses=3)
 

def velocity(prev_x,cur_x,t):
    """
    Function to calculate the velocity of the hand

    @param prev_x: Previous coordinate of hand
    @param cur_x: Current coordinate of hand
    @returns: The velocity of the hand
    """
    try:
     vel = (cur_x - prev_x)/t
     #if a zero oocurs at the denominator
    except ZeroDivisionError:
     return -1
    return vel
 #
def detect_motion(x1,y1,x2,y2,t):
    """
    Function to detect the action of hand on the basis of velocity

    @param x1,y1 : Previous coordinates of hand
    @param x2,y2 : Current coordinates of hand
    @param t : Total time elapsed between (x1,y1) & (x2,y2)
    @returns: The action that has been performed
    """
    vel_x = int(velocity(x1,x2,t))
    vel_y = int(velocity(y1,y2,t))
    print("Velocity of x is",vel_x)
    # print("Velocity of y is",vel_y)
    if vel_x > 120:
        # print("Hello")
        return MOTION_RIGHT
    elif vel_x < -120:
        return MOTION_LEFT
    # if vel_y > 30:
    #     return MOTION_DOWN
    # elif vel_y < -30:
    #     return MOTION_UP
    else:
        return NO_MOTION



frame_num = 1 # Counter for frame number
prev_x, prev_y, cur_x, cur_y = -1,-1,-1,-1 # Initializing the previous and current co-ordinates of the hand centroid
last_timestamp = 0 # Initializing the last recording timestamp 
mp_drawings = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands #Intializing the hand detection model from mediapipe
cap = cv2.VideoCapture(0) # Initializing the webcam 
print(cap.get(cv2.CAP_PROP_FPS))

with mp_hands.Hands(model_complexity = 1,min_detection_confidence = 0.4,min_tracking_confidence = 0.6,max_num_hands = 1) as hands :
    while cap.isOpened():
        success,image = cap.read()
        # cv2.imshow("frame",image)
        h,w,_ = image.shape
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) #Adjusting the contrast on the image
        image.flags.writeable = False
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results = hands.process(image) # Detecting the frames for hands
        # print(results.multi_handedness)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(f'Index finger Coordinates: (',f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * h},'f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * w}')
                if prev_x == -1:
                  prev_x,prev_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * h,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * w
                  frame_num = 1
                  print(prev_x,prev_y)
                if frame_num == 0:
                    cur_time = time.time()
                    time_elapsed = cur_time - last_timestamp
                    hand_motion = detect_motion(prev_x,prev_y,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*h,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * w,time_elapsed)
                    perfrom(hand_motion,prev_x,prev_y,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*h,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * w)
                    prev_x,prev_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x *h,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * w
                    print(hand_motion)
                    last_timestamp = time.time()
                    
                    # Re-initializing the frame counter
                    if hand_motion!=NO_MOTION:
                        frame_num=1
                    else:
                        frame_num=2
                else:
                    frame_num-=1

                # Drawing the landmarks and landmark connections on the frame            
                mp_drawings.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        else:
            prev_x =-1
            prev_y=-1
            frame_num = 4
        cv2.imshow("Hands",cv2.flip(image,1))
        #if esc key is pressed then stop the program
        if cv2.waitKey(5) & 0XFF == 27:
            break
cap.release()