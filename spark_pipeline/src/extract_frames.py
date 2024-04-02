import cv2
import sys
import base64
import json

def video_to_frames(input_loc, meta_data, output_loc):
    vidcap = cv2.VideoCapture(input_loc)
    success, image = vidcap.read()
    print(success)
    count = 0

    start_timestamp = meta_data["start_timestamp"]
    fps = meta_data["fps"]

    seconds_per_frame = 1.0/float(fps)

    with open(f"{output_loc}/meta.txt", "w") as meta_f, open(f"{output_loc}/all_frames64.csv", "w") as frames_f:
        frames_f.write(f"frame_num,frame_time,frame_data\n")
        while success:
            frame_time = start_timestamp + (count*seconds_per_frame)
            file_name = f"frame_{count}_{frame_time}"
            cv2.imwrite(f"{output_loc}/{file_name}.jpg", image)  
            meta_f.write(file_name+"\n")
            
            _, im_arr = cv2.imencode(".jpg", image)
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)
            frames_f.write(f"{count},{frame_time},{im_b64.decode('utf-8')}\n")
            
            success, image = vidcap.read()
            count += 1

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print("input_path:::", input_path)
    print("output_path:::", output_path)

    meta_data = json.loads(open(f"{input_path}/meta.json").read())
    
    video_to_frames(f"{input_path}/test_video.mp4", meta_data, output_path)
    