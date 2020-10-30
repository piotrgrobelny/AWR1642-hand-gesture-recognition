
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import Qt
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import read_data_awr1642 as read_awr
import time
import numpy as np

#gui template generated from qtdesigner
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(656, 900)
        Dialog.setStyleSheet("background-color: rgb(255, 255, 255);")
        font = QtGui.QFont()
        font.setPointSize(20)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.graphicsView = PlotWidget(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 621, 580))
        self.graphicsView.setObjectName("graphicsView")
        font = QtGui.QFont()
        font.setPointSize(20)

        #change number of gesture to open/save
        vbox = QtWidgets.QVBoxLayout()
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        vbox.addWidget(self.spinBox)
        self.spinBox.setGeometry(500, 600, 70, 51)
        self.spinBox.setValue(1)
        font1 =  QtGui.QFont()
        font1.setPointSize(20)
        self.spinBox.setFont(font1)

        #label
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(60, 820, 500, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")

        #buttons for saving data
        self.button1 = QtWidgets.QPushButton(Dialog)
        self.button1.setText("Start recording")
        self.button1.setGeometry(QtCore.QRect(60, 600, 130, 51))

        #button to play records from database
        self.button2 = QtWidgets.QPushButton(Dialog)
        self.button2.setText("Play the record")
        self.button2.setGeometry(QtCore.QRect(60, 700, 130, 51))

        #list to choose what gesture you want to write into .csv file
        self.list = QtWidgets.QListWidget(Dialog)
        self.list.setGeometry(QtCore.QRect(240, 600, 200, 230))
        self.list.addItem('Close fist horizontally')
        self.list.addItem('Close fist perpendicularly')
        self.list.addItem('Hand to left')
        self.list.addItem('Hand to right')
        self.list.addItem('Hand rotating palm up')
        self.list.addItem('Hand rotating palm down')
        self.list.addItem('Arm to left')
        self.list.addItem('Arm to right')
        self.list.addItem('Move hand closer to radar')
        self.list.addItem('Move hand away from radar')
        self.list.addItem('Hand up')
        self.list.addItem('Hand down')
        self.list.addItem('Stop gesture')
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "AWR1642 saving gestures application"))
        self.label.setText(_translate("Dialog", "Reading data...."))

class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.timer = QtCore.QTimer() #timer which counts time to update data, set for one 100ms
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start(35)
        self.ui.graphicsView.setBackground('w')
        self.ui.graphicsView.setXRange(-0.3, 0.3)
        self.ui.graphicsView.setYRange(0, 1)
        self.ui.graphicsView.setLabel('left', text='Y position (m)')
        self.ui.graphicsView.setLabel('bottom', text='X position (m)')

        self.data_plot = self.ui.graphicsView.plot([], [], pen=None, symbol='o')
        self.ui.graphicsView.setXRange(-0.5,0.5)
        self.ui.graphicsView.setYRange(0,1)

        self.ui.button1.setStyleSheet("background-color: rgb(63, 196, 84);")
        self.ui.button1.clicked.connect(self.button1_clicked)  # start recording button
        self.click_button1 = 0  # state of button1, 0 - not pushed, 1 - pushed

        self.ui.button2.setStyleSheet("background-color: rgb(63, 196, 84);")
        self.ui.button2.clicked.connect(self.button2_clicked) #play record from database button
        self.click_button2 = 0 # state of button2, 0 - not pushed, 1 - pushed

        self.ui.list.itemClicked.connect(self.list_clicked)
        self.filename = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", "hand_to_right",
                         "hand_rotation_palm_up","hand_rotation_palm_down", "arm_to_left", "arm_to_right",
                         "hand_closer", "hand_away", "hand_up", "hand_down", "stop_gesture"]
        self.onetime_event = 0

    def updatePlot(self): #update plot and values from radar
        if self.click_button2 == 0: #Plot data from AWR in real time when button has state 0
            self.data_plot.setData(read_awr.x, read_awr.y)
            QtGui.QApplication.processEvents()

        elif self.click_button2 == 1: #Plot record saved in database
            FrameData, FrameNumber = self.read_file() #Read data from csv file
            print("Start plot")

            for x in range(FrameNumber - 1):
                time.sleep(0.04)  # Slighlty slower rate due to delay inside read_data_awr1642 function
                detObj = FrameData[x]
                x = -detObj["x"]
                y = detObj["y"]
                try:
                    self.data_plot.setData(x, y)
                    QtGui.QApplication.processEvents()
                except:
                    print("error with plotting data from file")
            time.sleep(1)

    def button1_clicked(self): #Behavior of button1
        if self.click_button1 == 0 and self.click_button2 == 0: #Can't work if button2 is clicked
            self.ui.button1.setText("Stop recording") #Set text
            self.ui.button1.setStyleSheet("background-color: rgb(220, 82, 51);") #Set red colour
            self.click_button1 = 1 #Change state to 1 - clicked
            read_awr.if_button_is_pushed = 1 #Pass state of button to read_awr program

            if self.onetime_event == 0: #Pass the number of spinBox to GestureNumber just once at the start of program
                read_awr.GestureNumber = self.ui.spinBox.value()
                self.onetime_event = 1

            dirpath = self.filename[self.ui.list.currentRow()]
            path = "data/" + dirpath + "/gesture_" + str(read_awr.GestureNumber) + ".csv"
            self.label_text("Save as: " + path) #Set the label text when saving data

        elif self.click_button1 == 1 and self.click_button2 == 0:
            self.ui.button1.setText("Start recording")
            self.ui.button1.setStyleSheet("background-color: rgb(63, 196, 84);") #Set green colour
            self.click_button1 = 0
            read_awr.if_button_is_pushed = 0
            print("Number of frames: ", read_awr.FrameIndex)

    def button2_clicked(self): #Behavior of clicked button2
        if self.click_button2 == 0 and self.click_button1 == 0:
            self.ui.button2.setText("Stop playing")
            self.ui.button2.setStyleSheet("background-color: rgb(220, 240, 4);")
            print("Play gesture record from database")
            self.click_button2 = 1
        elif self.click_button2 == 1 and self.click_button1 == 0:
            self.ui.button2.setText("Play the record")
            self.ui.button2.setStyleSheet("background-color: rgb(63, 196, 84);")
            print("Back to real time plot")
            self.click_button2 = 0

    def list_clicked(self):
        row_number = self.ui.list.currentRow() #read the number of clicked row
        read_awr.filename = self.filename[row_number] #go to directory assigned to row

    def read_file(self):
        row_number = self.ui.list.currentRow() # read the number of clicked row
        gesture_number = self.ui.spinBox.value()
        dirpath = self.filename[row_number]

        print("Start reading data from csv file")

        path = "data/" + dirpath + "/gesture_" + str(gesture_number+100) + ".csv"
        self.label_text("Open: "+path)
        data = np.loadtxt(path, delimiter=",", skiprows=2) #open file

        FrameNumber = 1
        FrameData = {}

        for x in range(len(data)):
            if data[x][0] == FrameNumber:  # check if this is the same frame
                # init of variables - first row of the frame
                distance = data[x][2]
                velocity = data[x][3]
                peak_value = data[x][4]
                x_pos = data[x][5]
                y_pos = data[x][6]
                number_of_objects = int(data[x][1])
                try:
                    y = 0
                    while data[y + x + 1][0] == FrameNumber: #while the same frame
                        distance = np.append(distance, data[y + x + 1][2])
                        velocity = np.append(velocity, data[y + x + 1][3])
                        peak_value = np.append(x_pos, data[y + x + 1][4])
                        x_pos = np.append(x_pos, data[y + x + 1][5])
                        y_pos = np.append(y_pos, data[y + x + 1][6])
                        y += 1
                    detobj = {"frame": FrameNumber, "objectNumber": number_of_objects, "range": distance,
                              "velocity": velocity, "peakval": peak_value, "x": x_pos, "y": y_pos}
                    FrameData[FrameNumber - 1] = detobj
                    FrameNumber += 1
                except:
                    print("index out of bound")

        print("Close file")

        return FrameData, FrameNumber

    def label_text(self,text): #change label text
        self.ui.label.setText(text)

def plot():
    pg.setConfigOption('background', 'w')
    win = pg.GraphicsWindow(title="2D scatter plot")
    p = win.addPlot()
    p.setXRange(-0.5, 0.5)
    p.setYRange(0, 1)
    p.setLabel('left', text='Y position (m)')
    p.setLabel('bottom', text='X position (m)')
    s = p.plot([], [], pen=None, symbol='o')

if __name__ == "__main__":
    read_awr.main() #start read and process data from radars
    import sys
    app = QtWidgets.QApplication(sys.argv)
    c = MainWindow()
    c.show()
    sys.exit(app.exec_())