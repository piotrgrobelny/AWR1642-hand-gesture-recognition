import sys
import time
import traceback
from threading import Lock, Thread

import numpy as np
import serial
import serial.tools.list_ports

configFileName = '1642config.cfg'
move_lock = Lock()
hb100_data = ''
distance = 0
if_button_is_pushed = 0
filename = ""
x = []
y = []


def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 2

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

            digOutSampleRate = int(splitWords[11]);

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
            2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


def Read_DataAWR(Dataport, configParameters):
    print("###################################")
    print("Start reading data from AWR")
    global byteBuffer, byteBufferLength, hb100_data, slope, GestureNumber
    slope = 0  # detect falling slope for button
    GestureNumber = 1  # save with asceding order 'gesture1', 'gesture2' etc
    byteBuffer = np.zeros(2 ** 15, dtype='uint8')
    byteBufferLength = 0;
    # Constants
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    maxBufferSize = 2 ** 15;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    detObj = {}
    tlv_type = 0

    # Arrays to store gesture data for ML
    dtype = [('framenumber', np.int16), ('objnumber', np.int16), ('Range', np.float64),
             ('velocity', np.float64), ('peakval', np.float64), ('x', np.float64), ('y', np.float64)]

    global GestureData, FrameIndex
    GestureData = np.array([(0, 0, 0, 0, 0, 0, 0)], dtype=dtype)
    FrameIndex = 1

    while True:
        readBuffer = Dataport.read(Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (byteBufferLength + byteCount) < maxBufferSize:
            byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
            byteBufferLength = byteBufferLength + byteCount

        # Check that the buffer has some data
        if byteBufferLength > 16:

            # Check for all possible locations of the magic word
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = byteBuffer[loc:loc + 8]
                if np.all(check == magicWord):
                    startIdx.append(loc)

            # Check that startIdx is not empty
            if startIdx:

                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                    byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(
                        len(byteBuffer[byteBufferLength - startIdx[0]:]),
                        dtype='uint8')
                    byteBufferLength = byteBufferLength - startIdx[0]

                # Check that there have no errors with the byte buffer length
                if byteBufferLength < 0:
                    byteBufferLength = 0

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Read the total packet length
                totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

                # Check that all the packet has been read
                if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                    magicOK = 1

        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Initialize the pointer index
            idX = 0

            # Read the header
            magicNumber = byteBuffer[idX:idX + 8]
            idX += 8
            version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4

            # Read the TLV messages
            for tlvIdx in range(numTLVs):

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Check the header of the TLV message
                try:
                    tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
                    idX += 4
                    tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
                    idX += 4
                except:
                    pass

                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                    # word array to convert 4 bytes to a 16 bit number
                    word = [1, 2 ** 8]
                    tlv_numObj = np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2
                    tlv_xyzQFormat = 2 ** np.matmul(byteBuffer[idX:idX + 2], word)
                    idX += 2

                    # Initialize the arrays
                    rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                    dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                    peakVal = np.zeros(tlv_numObj, dtype='int16')
                    x = np.zeros(tlv_numObj, dtype='int16')
                    y = np.zeros(tlv_numObj, dtype='int16')
                    z = np.zeros(tlv_numObj, dtype='int16')

                    for objectNum in range(tlv_numObj):
                        # Read the data for each object
                        rangeIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        peakVal[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        x[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        y[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2
                        z[objectNum] = np.matmul(byteBuffer[idX:idX + 2], word)
                        idX += 2

                    # Make the necessary corrections and calculate the rest of the data
                    rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                    dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"] / 2 - 1)] = dopplerIdx[dopplerIdx > (
                            configParameters["numDopplerBins"] / 2 - 1)] - 65535
                    dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat

                    # Store the data in the detObj dictionary
                    detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                              "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                    # print(detObj)
            # Remove already processed data
            try:
                if idX > 0 and byteBufferLength > idX:
                    shiftSize = totalPacketLen

                    byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
                    byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                         dtype='uint8')
                    byteBufferLength = byteBufferLength - shiftSize

                    # Check that there are no errors with the buffer length
                    if byteBufferLength < 0:
                        byteBufferLength = 0
            except:
                pass

            update_plot(detObj)

            # store data in GestureData to save it in csv
            if if_button_is_pushed == 1:
                time.sleep(0.02)
                slope = 1  # to detect falling slope when clicking second time to save data
                try:
                    counter = 0
                    next_frame = 0
                    for i in range(len(detObj['x'])):
                        if detObj['x'][i] <= 0.3 and detObj['x'][i] >= -0.4: #threshold for x position
                            if detObj['y'][i] > 0 and detObj['y'][i] <= 0.6: #and detObj['y'][i] > 0.05: #threshold for y position
                                counter += 1
                                FrameData = np.array([(FrameIndex, counter, detObj['range'][i],
                                                       detObj['doppler'][i], detObj['peakVal'][i],
                                                        detObj['x'][i], detObj['y'][i])], dtype=dtype)
                                GestureData = np.append(GestureData, FrameData, axis=0)
                                next_frame = 1
                    if next_frame == 1:
                        FrameIndex += 1

                except Exception:
                    traceback.print_exc()


            if if_button_is_pushed == 0 and slope == 1:
                print("Save gesture to database")
                slope = 0
                dirpath = 'data/' + filename + '/gesture_' + str(GestureNumber+200) + '.csv'
                print(dirpath)
                np.savetxt(dirpath, GestureData, delimiter=',', fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f'],
                           header='FrameNumber, ObjectNumber,Range,Velocity,PeakValue,x,y', comments='')
                GestureData = np.array([(0, 0, 0, 0, 0, 0, 0)], dtype=dtype)
                GestureNumber += 1
                FrameIndex = 1


def update_plot(detObj):
    global x, y
    # write data to x,y lists, application_plotting will read them to update plot
    try:
        if len(detObj['x']) > 0:
            x = -detObj['x']
            y = detObj['y']
    except:
        pass


def configureSerialAWR():
    serial_port_1 = ''
    serial_port_2 = ''
    # Check which ports are used by awr1642
    ports = serial.tools.list_ports.comports()
    for port, desc, hwid in sorted(ports):
        if "Application/User" in desc:
            serial_port_1 = str(port)
        elif "Data" in desc:
            serial_port_2 = str(port)

    CLIport = serial.Serial(serial_port_1, 115200)
    if CLIport == None: sys.exit(1)

    Dataport = serial.Serial(serial_port_2, 921600)
    if Dataport == None: sys.exit(1)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)

    return Dataport


def limit_area(detObj):
    global distance
    # calculate distance from points in selected area
    try:
        a = detObj["y"]
        b = detObj["x"]

        limit_x = np.where(b > 0.15) and np.where(b < -0.15)
        a = np.delete(a, limit_x)

        limit_y = np.where(a > 1) or np.where(a < 0.02)
        a = np.delete(a, limit_y)
    except:
        pass
    try:
        distance = np.sort(a)[0]
    except:
        distance = 0

    return distance


def save_data(GestureData):
    np.savetxt('data/Gesture.csv', GestureData, delimiter=',', fmt=['%d', '%d', '%f', '%f', '%f', '%f'],
               header='FrameNumber, ObjectNumber,Range,Velocity,x,y', comments='')


def main():
    # make configuration before running threads
    configParameters = parseConfigFile(configFileName)

    p2 = Thread(target=Read_DataAWR, args=(configureSerialAWR(), configParameters))
    p2.start()


if __name__ == "__main__":
    main()
