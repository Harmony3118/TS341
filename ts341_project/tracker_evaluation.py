"""
# Input -----------------

- A list of the real position of the drone per frame for each video
- The list of videos
- yolo model (model = YOLO("best_yolo8s.pt"))
- the two trackers : 
    - simple_tracker = SimpleTrack(max_age=80)
    - imm_tracker = IMMByteTrack(max_age=100))

# Objects -----------------
- A named-tupple that will contain each detected position (X,Y) of the center of the drone for each frame, according to each tracker, for each video.
    Note on the videos : they should all have an ID, starting from 1.
- A list of the real position of the drone per frame for each video
- The total number of videos. 

# Algorithm -----------------

function user_choice : 
    ask to choose to display graphs
    if the user wants to choose :
        choose for which video he wants to see the results, thanks to their ID (given total number of video + 1)
        return the chosen video ID
    return 0

function compute_positions :
    for each video, 
        for each frame, 
            expected_position = the real position (X, Y) of the center of the drone

            for each tracker,
                detected_position = get the coordinates (X,Y) of the center of the drone according to the tracker
                rmse = np.sqrt(np.mean((expected_position - detected_position)**2, axis=0)) # compute the Root Mean Square Error

                add the detected position and the rmse to their respective tab
        
        for each tracker :
            compute the mean of the RMSE

    for each tracker : 
        compute the mean of the means of the RMSE

user_choice()
if the answer is not 0 :
    for the chosen video :
        plot the results in the same figure : 
            - The first subfigure on the top should display the deteced position of each tracker, according to each frame in absisca
            - The second subfigure on the bottom should display the RMSE of each tracker, according to each frame in absisca

for each tracker :
    print the mean RMSE
            
-> We should select the one for which the RMSE is the weakest.
    
# Output -----------------

- A graph of position (X, Y) comparison, giving for each frame : 
    - The expected (X, Y) position of the drone on the frame
    - The (X, Y) position givent by each tracker

- The RMSE between each tracker and the expected position, for each frame.

"""