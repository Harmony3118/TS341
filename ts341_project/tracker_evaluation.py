import logging


def try_open(f_name: str):
    f_truth = None
    try:
        f_truth = open(f_name)
    except OSError:
        logging.error(f"File not found : {f_name}")
        exit(1)
    return f_truth


def main():
    from yolo_plus_imm import video_to_json_output
    import json
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
    data_truth_filename = filedialog.askopenfilename()
    f_truth = try_open(data_truth_filename)
    truth = np.array(json.loads("".join(f_truth.readlines())))

    data_yolo_filename = video_to_json_output(ref_video)
    f_yolo = try_open(data_yolo_filename)
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
    diff = np.sqrt(np.linalg.norm(np.power(yolo, 2) - np.power(truth, 2), axis=1))

    f_truth.close()
    f_yolo.close()
    logging.info(f"RMSE: {np.sqrt(np.sum(diff) / len(diff))}")
    plt.plot(np.arange(len(yolo)), diff)
    plt.xlabel("Frame k")
    plt.ylabel("Pixel distance error (normalized)")
    plt.title(
        "Squared error accuracy in position between YOLO + tracker and ground truth"
    )
    plt.show()
