# TS-341 Project

This project proposes a drone detection algorithm. The current solution follows these 3 steps :

1. **Drone detection** thanks to YOLO, an AI made for object detection
2. Adjustment of the **estimation of the object position** on each frame, thanks to a tracker based on Kalman filter
3. Computation of the drone **position in the world**, taking the camera as the center.

## How to use the drone detection system

### Quick setup

#### Repository

Clone this repo on your own machine. You can use an editor such as Visual Studio Code, for instance.

```
https://github.com/Harmony3118/TS341.git
```

#### Videos

Place all your videos into the _"videos"_ folder. Name them with something short or easy to type, as the program will ask you to enter which video you would like to use.

**NB :** a video name should not include any whitespace.

### Run the drone detection script

Once you have opened the project in your editor, you can follow these steps directly in the terminal.

1. Enter the right **folder**

   ```
   cd ts341_project
   ```

2. Run the **program**

   ```
   poetry install
   poetry run yolo
   ```

3. Execute the instructions in the console:
   - Choose a video;
   - Choose a camera;
   - Specify a YOLO model;
   - Choose to display or not the video.

After this, a **new window** will open and you will see the video. If the drone is **detected**, a detection point will be displayed on the video, and its **3D coordinates** will be printed in the console.

### Quit

#### Video display ON

Press '**q**' on the video window.

#### Video display OFF

Press '**Ctrl+C**' in the editor terminal.

## How does it work?

This algorithm is described in the project's `yolo_plus_imm.py` file. It has several sections.

1. **User selection:**

   a. Video loading. The user is asked which file they want to select. Videos must be placed in the project's “videos” folder and named without spaces.

   b. Camera specification. The user must specify the model of camera used to film the selected video. The type of camera influences the calculation of the drone's position in space.

   c. YOLO model specification. We have fine-tuned a model that is therefore recommended, but it is possible to use the default model to compare performance.

   d. Video display. Can be disabled to use fewer resources.

2. **Calculation of the focal length** from the camera settings, as well as the centre of the image, which will be the origin for the 3D reference point centred around the camera.

3. **Retrieval of the drone's position** in the image for each frame of the video. This is made possible by YOLO, potentially assisted by the tracker, which attempts to keep the position prediction active. The track is kept if its confidence score is above 60%. The tracker used in adition is Bytetrack, a popular solution for accurate object tracking.

4. **Estimation of the drone's 3D position** around the camera. The distance between the camera and the drone is calculated by dividing the actual height of the drone by its height on the frame, which is influenced by the focal length value.

5. **Display of detections** on the video and coordinates in the console.

## Heavy files

The model data and weights are already available in the repository. As for the videos, feel free to paste yours into the _"videos"_ folder, once you cloned the repository on your own machine.

## About outputs
