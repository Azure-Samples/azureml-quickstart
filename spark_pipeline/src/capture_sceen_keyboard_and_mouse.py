# Import necessary modules
import cv2
import numpy as np
import pyautogui
import threading
import mouse
import keyboard
from datetime import datetime
from time import sleep
import json
import sys
import os

mouse_events = []
keyboard_events = []

def keyboard_event_hook(event):
    keyboard_events.append((event.time, event.to_json()))

def capture_keyboard_and_mouse(secs):

    mouse.hook(mouse_events.append)
    keyboard.hook(keyboard_event_hook)
    keyboard.start_recording()  
    for i in range(secs):
        print(f"Collecting input: {i} of {secs}")
        sleep(1)


def capture_sceen_keyboard_and_mouse(output_dir, record_seconds = 30, fps = 12.0, start_timestamp = datetime.now().timestamp()):
    # Initialize video parameters
    SCREEN_SIZE = tuple(pyautogui.size())
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 12.0
    #record_seconds = 30

    # Create the video writer object
    out = cv2.VideoWriter(f"{output_dir}/test_video.mp4", fourcc, fps, SCREEN_SIZE)

    frames_to_capture = int(record_seconds * fps)

    input_capture_thread = threading.Thread(target=capture_keyboard_and_mouse, args=[record_seconds])
    input_capture_thread.start()

    # Main loop for capturing screenshots and writing frames
    for i in range(frames_to_capture):
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        #cv2.imshow("screenshot", frame)
        #if cv2.waitKey(1) == ord("q"):
        #    break

    # Clean up
    out.release()
    cv2.destroyAllWindows()

    input_capture_thread.join()
    print("Input collection finished!")
    mouse.unhook(mouse_events.append)
    keyboard.unhook(keyboard_event_hook)
    keyboard.stop_recording()

if __name__ == "__main__":
    print("Capturing screen, keyboard and mouse...", sys.argv[1])
    #output_dir = "../output_directory/test_video" #sys.argv[2]

    start_timestamp = datetime.now().timestamp()

    output_dir = f"../output_directory/capture_{start_timestamp}" #sys.argv[2]
    os.makedirs(output_dir)

    record_seconds = int(sys.argv[1])
    fps = 12.0 
     
    capture_sceen_keyboard_and_mouse(output_dir, record_seconds=record_seconds, fps=fps, start_timestamp=start_timestamp)
    print("Done capturing!")
    
    with open(f"{output_dir}/mouse_events.csv", "w") as f_mouse, open(f"{output_dir}/button_events.csv", "w") as f_button:
        mouse_col_names = ["time","X","Y"]
        button_col_names = ["time","event_type","button"]
        
        f_mouse.write(",".join(mouse_col_names)+"\n")
        f_button.write(",".join(button_col_names)+"\n")
        
        for event in mouse_events:
            #print("events:", event)
            event_type = type(event).__name__
            if event_type == "MoveEvent":
                s = f"{event.time},{event.x},{event.y}"
                f = f_mouse
            else:
                s = f"{event.time},{event.event_type},{event.button}"
                f = f_button
            f.write(s+"\n")

    #print(mouse_events)
    with open(f"{output_dir}/keyboard_events.csv", "w") as f:

        col_names = ["time","event_type","scan_code","name","is_keypad"]
        
        f.write(",".join(col_names)+"\n")

        for event in keyboard_events:
            #print(event)
            event_obj = json.loads(event[1])
            event_data = ",".join(str(event_obj[key]) for key in col_names)
            s = f"{event_data}"
            f.write(s+"\n")

    with open(f"{output_dir}/meta.json", "w") as meta_f:
        meta_f.write(json.dumps({"start_timestamp" : start_timestamp,
                                 "record_seconds" : record_seconds,
                                 "fps" : fps}))