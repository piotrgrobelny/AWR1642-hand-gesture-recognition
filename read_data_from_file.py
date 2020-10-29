import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import traceback

def read_file():
    print("Start reading data from csv file")

    data = np.loadtxt("data/close_fist/gesture_3.csv", delimiter=",", skiprows=2)
    FrameNumber = 1
    FrameData = {}
    Threshold = 1

    for x in range(len(data)):
        if data[x][0] == FrameNumber: # check if this is the same frame
            #init of variables
            distance = data[x][2]
            velocity = data[x][3]
            peak_value = data[x][4]
            x_pos = data[x][5]
            y_pos = data[x][6]
            number_of_objects = int(data[x][1])
            try:
                for y in range(number_of_objects-1):  # check number of objects in frame
                    peak = data[y + x + 1][4]
                    if peak > Threshold:
                        distance = np.append(distance, data[y + x+1][2])
                        velocity = np.append(velocity, data[y + x+1][3])
                        peak_value = np.append(x_pos, data[y + x + 1][4])
                        x_pos = np.append(x_pos, data[y + x+1][5])
                        y_pos = np.append(y_pos, data[y + x+1][6])
                detobj = {"frame": FrameNumber, "objectNumber": number_of_objects, "range": distance,
                          "velocity": velocity, "peakval": peak_value, "x": x_pos, "y": y_pos}
                FrameData[FrameNumber - 1] = detobj
                FrameNumber += 1
            except Exception:
                traceback.print_exc()

    print("Close file")

    return FrameData, FrameNumber

def update():
    FrameData, FrameNumber = read_file()
    print("start plot")
    for x in range(FrameNumber-1):
        time.sleep(0.04) #slighlty slower rate due to delay inside read_data_awr1642 function
        detObj = FrameData[x]
        #print(detObj["frame"])
        x = -detObj["x"]
        y = detObj["y"]

        try:
            s.setData(x, y)
            QtGui.QApplication.processEvents()
        except Exception:
            traceback.print_exc()

# START QtAPPfor the plot
app = QtGui.QApplication([])

# Set the plot
pg.setConfigOption('background', 'w')
win = pg.GraphicsWindow(title="2D scatter plot")
p = win.addPlot()
p.setXRange(-0.5, 0.5)
p.setYRange(0, 1.5)
p.setLabel('left', text='Y position (m)')
p.setLabel('bottom', text='X position (m)')
s = p.plot([], [], pen=None, symbol='o')

def main():
    update()

if __name__ == "__main__":
    main()