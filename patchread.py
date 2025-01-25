import cv2
import numpy as np
import tkinter as tk
def read_video(video_path,roi,imager):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    pt_means, pt_stds = [], []
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        frames.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break
        pt_means.append(get_patch(frame, imager["gr_coord"], roi)[0])
        pt_stds.append(get_patch(frame, imager["gr_coord"], roi)[1])
        # Resize the frame by a factor of 2
        resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        # Display the resulting frame
        cv2.imshow('Frame', resized_frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    plot(pt_means,frames, pt_stds)

def plot(pt_means, frames, pt_stds):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(frames[:-1], pt_means, label="Mean")
    plt.xlabel('Frames')
    plt.ylim(0, 265)
    plt.ylabel('Mean')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(frames[:-1], pt_stds, label="Std")
    plt.xlabel('Frames')
    plt.ylabel('Std')
    plt.ylim(0, 80)
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_patch(frame, coord, roi):
    x, y = coord
    patch = frame[y - roi:y + roi, x - roi:x + roi]
    return np.mean(patch), np.std(patch)

if __name__ == "__main__":

    video_path = tk.fileddialog()
    roi = 30
    imx678 = {"wh_coord": (1506, 1068), "gr_coord": (1456, 1090), "bl_coord": (1511, 1262)}
    ar0234 = {"wh_coord": (791, 564), "gr_coord": (757, 565), "bl_coord": (792, 546)}
    read_video(video_path, roi, imager=imx678)