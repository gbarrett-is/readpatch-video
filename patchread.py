import cv2
import numpy as np
import tkinter as tk
import tkinter.filedialog as fd

def read_video(video_path,roi):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    pt_means, pt_stds = [], []
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        frames.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
        pt_means.append(get_patch(frame, roi)[0])
        pt_stds.append(get_patch(frame, roi)[1])
        
        # Resize the frame by a factor of 2
        resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.rectangle(resized_frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Frame', resized_frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    plot(pt_means,frames, pt_stds)

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        if 'roi_points' not in param:
            param['roi_points'] = []
        param['roi_points'].append((x, y))

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

def get_patch(frame, roi):
    #TODO: Fix this: the patch is not being extracted correctly returns a nan value. probably and empty slice
    x1, y1 = roi[0],roi[1]
    x2, y2 = roi[0:1] + roi[2:3]
    patch = frame[y1:y2, vc x1:x2]
    return np.mean(patch), np.std(patch)

def define_roi(frame):
    roi = []
    param = {}
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_click, param=param)
    while True:
        frame_sz = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow('Frame', frame_sz)
        if len(param.get('roi_points', [])) == 2:
            x1, y1 = param['roi_points'][0]
            x2, y2 = param['roi_points'][1]
            roi.extend([min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)])
            print(f"ROI selected: {roi}")
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    return roi

def get_nth_frame(video_path, n):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {n}.")
        return None

    return frame

def main():
    frame_n = 15
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = fd.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mkv")])
    #get_nth_frame(video_path, frame_n)
    roi = define_roi(get_nth_frame(video_path, frame_n))
    read_video(video_path,roi)
if __name__ == "__main__":
    main()
   #imx678 = {"wh_coord": (1506, 1068), "gr_coord": (1456, 1090), "bl_coord": (1511, 1262)}
    #ar0234 = {"wh_coord": (791, 564), "gr_coord": (757, 565), "bl_coord": (792, 546)}