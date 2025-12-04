## How to use the drone detection system

### Quick setup

#### Repo

Clone this repo on your own machine. You can use an editor such as Visual Studio Code, for instance.

```
https://github.com/Harmony3118/TS341.git
```

#### Videos

Place all your videos into the _"videos"_ folder. Name them with something short or easy to type, because the program will ask you which video you would like to use.

### Run the drone detection script

Once you have opened the project in your editor, you can follow these steps directly in the console.

1. Enter the right **folder**

   ```
   cd ts341_project
   ```

2. Run the **program**

   ```
   poetry run yolo
   ```

3. Execute the instructions in the console. First, choose a video, and then the corresponding camera.

After this, a **new window** will open and you will see the video. If the drone is **detected**, a detection point will be displayed on the video, and its **3D coordinates** will be printed in the console.

### Quit

Press '**q**'.
