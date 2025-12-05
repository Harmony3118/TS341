def main():
    from yolo_plus_imm import video_to_json_output
    import json
    import logging
    from tkinter import filedialog
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt

    print("Select reference video")
    ref_video = filedialog.askopenfilename()
    vc = cv.VideoCapture(ref_video)
    ret, frame = vc.read()
    if not ret or not vc.isOpened():
        logging.error("Could not open video file {}".format(ref_video))
    og_shape = frame.shape

    # read center of tracker data, truth and yolo
    print("Select labeled positions data JSON file")
    data_truth_filename = video_to_json_output(ref_video)
    f_truth = None
    try:
        f_truth = open(data_truth_filename)
    except OSError:
        logging.error(f"File not found : {data_truth_filename}")
        exit(1)

    truth = np.array(json.loads("".join(f_truth.readlines())))
    print("Select YOLO-estimated positions data JSON file")
    data_yolo_filename = filedialog.askopenfilename()
    f_yolo = open(data_yolo_filename)
    yolo = np.array(json.loads("".join(f_yolo.readlines())))

    # re-scale ground truth data
    # it was captured on a (720, 480) frame
    # original capture size is
    # this phase isn't required if the ground truth data was correctly obtained
    print(f"truth: {truth[0]} | yolo: {yolo[0]}")
    truth[:, 0] = (og_shape[0] * truth[:, 0]) / 720
    truth[:, 1] = (og_shape[1] * truth[:, 1]) / 480
    print(f"RESHAPED : truth: {truth[0]} | yolo: {yolo[0]}")

    # yolo data runs on the whole video
    # labeled data does not (because we were lazy labeling)
    # trim yolo data
    yolo = yolo[: len(truth)]
    assert (
        len(yolo) != 0
    ), "Error while trimming YOLO tracker data : truth data array empty"

    # compute error to truth
    diff = np.power(np.linalg.norm(yolo - truth, axis=1), 2)

    f_truth.close()
    f_yolo.close()
    print(f"RMSE: {np.sqrt(np.sum(diff) / len(diff))}")
    plt.plot(np.arange(len(yolo)), diff)
    plt.xlabel("Frame k")
    plt.ylabel("Pixel distance error (normalized)")
    plt.title(
        "Squared error accuracy in position between YOLO + tracker and ground truth"
    )
    plt.show()
