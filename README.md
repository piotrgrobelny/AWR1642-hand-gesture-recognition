# AWR1642-hand-gesture-recognition
Hand gesture recognition for AWR1642 BOOST evaluation board using LSTM classificaiton algorithm. The repository consist of two separate programs: application with GUI for saving the gesture into `CSV` file
inside `Data` directory and LSTM classification algorithm. The following code and database was used and described in this [MDPI article](https://www.mdpi.com/2079-9292/11/5/787). 

## Saving the gesture to database
Start configuration of AWR1642 boost is saved in `1642config.cfg` file. Connect the AWR1642 BOOST board to PC. Install neccesery libraries in order to run the python files.
Install the [mm-Wave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/4.2.0/) from Texas Instrument.
Run the `application.py` file:

![Alt Text](https://media.giphy.com/media/zkyNErcqgLfXrr8Fy4/giphy.gif)

### Recording mode
Applications shows detected targets by AWR1642 radar in real time along X and Y coordinates.Choose which gesture you want to record from the center list. Click the `Start Recording` button in order to start recording and click it again to stop the whole process.
Gesture will be saved into path shown at the bottom of the window.

### Preview mode
To preview the recorded file choose the type of gesture which you want to play from center list and number of the recorded sample from the right window. Click the `Play the record` button. X and Y grid will show the extracted gesture from the database.

### Database
`data` folder consist of 4600 samples from 4 persons.

## Classification algorithm
Open the `classification_algorithm.py` file. Choose training parameters: 
```
layers_number = 5
nodes_array = [32]
epochs = 10 #number of epochs
optimizer_step_value = 0.001 #optimizer step value for pytorch
test_percent = 0.1 #what percentage of the database will be used for the set test
parameters = 3 #how many input parameters of gestures involve in training
break_limit = 95 #stop after reaching this accuracy in %
```

Choose which gesture types are involved in training. In default there are 12 types:
```
LABELS = {ARM_TO_LEFT: 0, ARM_TO_RIGHT: 1,
              HAND_AWAY:2,  HAND_CLOSE:3,
              FIST_HORIZONTALLY: 4, FIST_PERPENDICULARLY: 5,
              HAND_RIGHT:6, HAND_LEFT:7,
              PALM_DOWN:8, PALM_UP:9,
              HAND_UP: 10, HAND_DOWN: 11
              }
```

`read_database` and `load_data` functions extract the gestures from `data` directory and save them into numpy file `training_data.npy`. AWR1642 collects the following parameters about the targets: velocity, peak value, x and y position and range.
You can choose which input parameters will be involved in training here:
```
 #############Choosing paramaters to extract########################
                framedata = np.array([x_pos, y_pos, velocity])
 ###################################################################
```
`start_ml` function starts the training process. There are multiple classification algorithm classes to work with: RNN, GRU, LSTM, LSTM+linear, GRU+linear and RNN+linear. I obtained best results with pure LSTM. Visualisation of Accuracy and Loss are handled by matplotlib:
```
# visualization loss
    plt.plot(iteration_list, loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Reccurent NN: Loss vs Number of iteration")
    plt.show()

    # visualization accuracy
    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("Reccurent NN: Accuracy vs Number of iteration")
    plt.show()
 ```

