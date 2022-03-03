import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pytorch version --",torch.__version__)

class Gestures():

    #Directories
    ARM_TO_LEFT = "arm_to_left"
    ARM_TO_RIGHT = "arm_to_right"
    FIST_HORIZONTALLY = "close_fist_horizontally"
    FIST_PERPENDICULARLY = "close_fist_perpendicularly"
    HAND_CLOSE = "hand_closer"
    HAND_AWAY = "hand_away"
    HAND_LEFT = "hand_to_left"
    HAND_RIGHT = "hand_to_right"
    HAND_DOWN = "hand_down"
    HAND_UP = "hand_up"
    PALM_DOWN = "hand_rotation_palm_down"
    PALM_UP = "hand_rotation_palm_up"
    STOP_GESTURE = "stop_gesture"

    LABELS = {ARM_TO_LEFT: 0, ARM_TO_RIGHT: 1,
              HAND_AWAY:2,  HAND_CLOSE:3,
              FIST_HORIZONTALLY: 4, FIST_PERPENDICULARLY: 5,
              HAND_RIGHT:6, HAND_LEFT:7,
              PALM_DOWN:8, PALM_UP:9,
              HAND_UP: 10, HAND_DOWN: 11

              }

    global class_number
    class_number = len(LABELS)
    training_data = []

    def read_database(self, dir):
        dataset = []
        dirpath = "data/" + dir + "/"

        for gesture in os.listdir(dirpath):
            path = dirpath + gesture
            data = np.loadtxt(path, delimiter=",", skiprows=2)  # skip header and null point

            FrameNumber = 1   # counter for frames
            pointlenght = 80  # maximum number of points in array
            framelenght = 80  # maximum number of frames in array
            datalenght = int(len(data))
            gesturedata = np.zeros((framelenght, frame_parameters, pointlenght))
            counter = 0

            while counter < datalenght:
                velocity = np.zeros(pointlenght)
                peak_val = np.zeros(pointlenght)
                x_pos = np.zeros(pointlenght)
                y_pos = np.zeros(pointlenght)
                object_number = np.zeros(pointlenght)
                iterator = 0

                try:
                    while data[counter][0] == FrameNumber:
                        object_number = data[counter][1]
                        range = data[counter][2]
                        velocity[iterator] = data[counter][3]
                        peak_val[iterator] = data[counter][4]
                        x_pos[iterator] = data[counter][5]
                        y_pos[iterator] = data[counter][6]
                        iterator += 1
                        counter += 1
                except:
                    pass

                #############Choosing paramaters to extract########################
                framedata = np.array([x_pos, y_pos, velocity])
                ###################################################################

                try:
                    gesturedata[FrameNumber - 1] = framedata
                except:
                    pass
                FrameNumber += 1

            dataset.append(gesturedata)
            number_of_samples = len(dataset)

        return dataset, number_of_samples

    def load_data(self):
        total = 0
        for label in self.LABELS:
            trainset, number_of_samples = self.read_database(label)

            for data in trainset:
                self.training_data.append([np.array(data),np.array([self.LABELS[label]])]) #save data and assign label

            total = total + number_of_samples
            print(label,number_of_samples)

        print("Total number:", total)

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.rnn = nn.RNN(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out,_ = self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)

        out = self.fc1(out)

        return out

class RNN_GRU(nn.Module):
    def __init__(self):
        super(RNN_GRU, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.gru = nn.GRU(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out,_ = self.gru(x,h0)
        out = out.reshape(out.shape[0],-1)

        out = self.fc1(out)

        return out

class RNN_LSTM(nn.Module):
    def __init__(self):
        super(RNN_LSTM, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.lstm = nn.LSTM(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out

class LSTM_LIN(nn.Module):
    def __init__(self):
        super(LSTM_LIN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.lstm = nn.LSTM(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

class GRU_LIN(nn.Module):
    def __init__(self):
        super(GRU_LIN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.gru = nn.GRU(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

class RNN_LIN(nn.Module):
    def __init__(self):
        super(RNN_LIN, self).__init__()

        self.hidden_size = neurons_num
        self.num_layers = num_layers
        self.rnn = nn.RNN(80 * frame_parameters, neurons_num, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(neurons_num * 80, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, class_number)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out

def start_ml(layer, nodes_number, epochs, optimizer_step_value, VAL_PCT, parameters, break_limit):
    global neurons_num
    global num_layers
    global frame_parameters

    num_layers = layer
    neurons_num = nodes_number
    frame_parameters = parameters

    net = RNN_LSTM().to(device)
    gestures = Gestures()
    gestures.load_data()

    #Load data from .npy
    dataset = np.load("training_data.npy", allow_pickle=True)

    #Convert from numpy to torch
    X = torch.Tensor([i[0] for i in dataset])
    y = torch.Tensor([i[1] for i in dataset])

    #Divide dataset to training set and test set
    val_size = int(len(dataset) * VAL_PCT)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    print("Trainingset:",len(train_X))
    print("Testset", len(test_X))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=optimizer_step_value)


    loss_list = []
    accuracy_list = []
    iteration_list = []
    tot_accuracy = 0
    count = 0


    print("Layers: ", layer, "Neurons: ", neurons_num)
    for epoch in range(epochs): # 3 full passes over the data
        for i in range(len(train_X)):

            data = train_X[i].to(device=device)
            targets = train_y[i].to(device=device)
            targets = torch.tensor(targets, dtype=torch.long)

            scores = net(data.view(-1, 80, 80 * frame_parameters))
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()
            count += 1

            tot_number = 0
            tot_correct = 0
            #Every 10 iterations calculate accuracy
            if count % 200 == 0:
                with torch.no_grad():

                    # Calculate accuracy for each gesture and total accuracy
                    for label in Gestures.LABELS:
                        correct = 0
                        number = 0
                        for a in range(len(test_X)):
                            if test_y[a] == Gestures.LABELS[label]:
                                X = test_X[a].to(device=device)
                                y = test_y[a].to(device=device)

                                output = net(X.view(-1, 80, 80 * frame_parameters))
                                for idx, i in enumerate(output):
                                    if torch.argmax(i) == y[idx]:
                                        tot_correct += 1
                                        correct += 1
                                    number += 1
                                    tot_number += 1

                    tot_accuracy = round(tot_correct / tot_number, 3)

                loss = loss.data
                loss = loss.cpu()
                loss_list.append(loss)
                iteration_list.append(count)
                accuracy_list.append(tot_accuracy)

        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss, tot_accuracy*100))
        if tot_accuracy*100>break_limit:
            break


    with torch.no_grad():
        #Calculate accuracy for each gesture and total accuracy
        tot_number = 0
        tot_correct = 0
        tot_propability = np.zeros([class_number, class_number + 1]) #propability for every output
        tot_classified = np.zeros([class_number, class_number + 1]) #Predicted labels matrix
        for label in Gestures.LABELS:
            correct = 0
            number = 0
            for a in range(len(test_X)):
                if test_y[a] == Gestures.LABELS[label]:
                    X = test_X[a].to(device=device)
                    y = test_y[a].to(device=device)
                    output = net(X.view(-1, 80, 80 * frame_parameters))
                    #Calculate propability distribution for each gesture
                    out = F.softmax(output, dim=1)
                    #copy memory back to CPU from GPU
                    out = out.cpu()
                    prob = out.numpy()*100
                    #print(prob, prob.shape)
                    prob = np.append(prob, 0)

                    y = y.cpu()
                    it = int(y.numpy())
                    tot_propability[it] = tot_propability[it] + prob
                    #last cell is the number of samples
                    tot_propability[it][class_number] = tot_propability[it][class_number] + 1

                    for idx, i in enumerate(output):
                        gesture_label_num = torch.argmax(i)
                        tot_classified[it][gesture_label_num] += 1
                        if torch.argmax(i) == y[idx]:
                            tot_correct += 1
                            correct += 1
                        number += 1
                        tot_number += 1

            print(label+" accuracy: ", round(correct/number, 3)*100)
        tot_accuracy = round(tot_correct / tot_number, 3)*100
        print("Total accuracy: ", tot_accuracy)
        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:16.1f}'.format})

        #save results
        with open("results.txt", 'a') as f:
                print(tot_accuracy, file=f)

        for i in range(class_number):
            #Saving confusion matrix

            print("Gesture", i)
            gest_num = int(tot_propability[i][class_number])
            tot_propability[i] = np.round((tot_propability[i]/gest_num),3)
            #print(tot_propability[i], gest_num)
            print(tot_classified[i], gest_num)
        #print("save txt")
        np.savetxt("propability matrix.txt", tot_propability, fmt='%f')
        np.savetxt("confusion matrix.txt", tot_classified, fmt='%f')

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


layers_number = 5
nodes_array = [32]
epochs = 10 #number of epochs
optimizer_step_value = 0.001 #optimizer step value for pytorch
test_percent = 0.1 #what percentage of the database will be used for the set test
parameters = 3
break_limit = 95
for layer in range(2, 3):
    for a in range(len(nodes_array)):
        nodes_number = nodes_array[a]
        start_ml(layer, nodes_number, epochs, optimizer_step_value, test_percent, parameters, break_limit)