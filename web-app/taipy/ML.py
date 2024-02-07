import math

import tensorflow as tf
from keras import layers, models
from keras import backend as K
import pickle
from matplotlib import pyplot as plt
import xgboost as xgb
from scipy import signal
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import mne

<<<<<<< Updated upstream
xgbf_file_name = "/Users/tapic/Documents/GitHub/Datathon2024/web-app/taipy/pkl/xgb_fd.pkl"
CNN_file_name = "/Users/tapic/Documents/GitHub/Datathon2024/web-app/taipy/pkl/cnn_td.pkl"
DNN_file_name = "/Users/tapic/Documents/GitHub/Datathon2024/web-app/taipy/pkl/dnn_td.pkl"
xgb_file_name = "/Users/tapic/Documents/GitHub/Datathon2024/web-app/taipy/pkl/xgb_td.pkl"
=======
xgb_file_name = "xgb_td.pkl"
xgbf_file_name = "xgb_fd.pkl"
CNN_file_name = "cnn_td.pkl"
DNN_file_name = "dnn_td.pkl"
>>>>>>> Stashed changes
# globally define training and test data paths
file_paths = [
    '/Training/p00_n1',
    '/Training/p01_n1',
    '/Training/p02_n1',
    '/Training/p02_n2',
    '/Training/p03_n1',
    '/Training/p03_n2',
    '/Training/p04_n1',
    '/Training/p04_n2',
    '/Training/p05_n1',
    '/Training/p05_n2',
    '/Training/p06_n1',
    '/Training/p06_n2',
    '/Training/p07_n1',
    '/Training/p07_n2',
    '/Training/p08_n1',
    '/Training/p08_n2',
    # '/Training/p11_n1',
    # '/Training/p11_n2',
    '/Training/p12_n2',
    '/Training/p13_n1',
    '/Training/p14_n1',
    '/Training/p14_n2',
    '/Training/p15_n1',
    '/Training/p15_n2',
    '/Training/p16_n1',
    '/Training/p16_n2',
    '/Training/p17_n1',
    '/Training/p19_n1',
    '/Training/p19_n2',
    '/Training/p20_n1',
    '/Training/p20_n2',
    '/Training/p21_n1',
    '/Training/p21_n2',
    '/Training/p22_n1',
    '/Training/p22_n2'
    # Add paths for all arrays
]

test_file_paths = [
    '/Training/p00_n2',
    '/Training/p09_n1',
    '/Training/p17_n2',
    '/Training/p18_n1',
    '/Training/p18_n2',
    '/Training/p22_n1',
    '/Training/p22_n2'
    # '/Training/p10_n1',
    #    '/Training/p10_n2'
]
CNN = models.Sequential()
DNN = models.Sequential()


def get_EEGs(patient_ndarray):
    return patient_ndarray[:, :2, :]



def call_CNN(file_name):
    X_test, _ = load_files([file_name], decimate=False, test=True)
    CNN = pickle.load(open(CNN_file_name, "rb"))
    y_pred = CNN.predict(X_test)
<<<<<<< Updated upstream
    return y_pred.transpose()
=======
    return y_pred
>>>>>>> Stashed changes


def call_DNN(file_name):
    X_test, _ = load_files([file_name], decimate=False, test=True)
    DNN = pickle.load(open(DNN_file_name, "rb"))
    y_pred = DNN.predict(X_test)
<<<<<<< Updated upstream
    return y_pred.transpose()
=======
    return y_pred
>>>>>>> Stashed changes


def call_FRQ(file_name):
    X_test, _ = load_freq_files([file_name], decimate=False,  test=True)
    xgb_model_f = pickle.load(open(xgbf_file_name, "rb"))
    xg_test = xgb.DMatrix(X_test)
    y_pred = xgb_model_f.predict(xg_test)
<<<<<<< Updated upstream
    # set any non zero values with zeros on either side to zero
    for i in range(2, len(y_pred) - 2):
        if y_pred[i] != 0 and y_pred[i - 1] == 0 and y_pred[i + 1] == 0:
            y_pred[i] = 0
=======
>>>>>>> Stashed changes
    return y_pred


def power_matrix(EEG_study):
    epochs = EEG_study.shape[0]
    matrix = np.zeros((5, epochs))
    for epoch in range(epochs):
        alpha = 0
        beta = 0
        delta = 0
        theta = 0
        sigma = 0

        data = EEG_study[epoch, :, :]
        sfreq = 100
        channel_names = ['Fpz-Cz', 'Pz-Oz']
        channel_types = ['eeg'] * 2
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        freqs, psd = signal.welch(data, fs=raw.info['sfreq'])

        # Down: Alpha, beta, delta, theta, sigma
        # Across: Each epoch's power vector: summarizing for one study in the night
        # Alpha processing: 8 to 12 Hz, Channel 1 (back of head)
        # Beta processing: 13 to 30Hz, Channel 0
        # Delta processing: 0.5 to 4Hz, Channel 0
        # Theta processing: 4 to 7Hz, Channel 0
        # Sigma processing: 12 to 16Hz, Channel 0

        for alpha_sample in range(21, 31):
            alpha += psd[1][alpha_sample]

        for beta_sample in range(34, 77):
            beta += psd[0][beta_sample]

        for delta_sample in range(2, 11):
            delta += psd[0][delta_sample]

        for theta_sample in range(11, 18):
            theta += psd[0][theta_sample]

        for sigma_sample in range(31, 41):
            sigma += psd[0][sigma_sample]

        # magnitude of final vector
        magnitude = math.sqrt(alpha ** 2 + beta ** 2 + delta ** 2 + theta ** 2 + sigma ** 2)

        matrix[2, epoch] = delta / magnitude
        matrix[1, epoch] = beta / magnitude
        matrix[3, epoch] = theta / magnitude
        matrix[0, epoch] = alpha / magnitude
        matrix[4, epoch] = sigma / magnitude

    return matrix


def load_freq_files(file_paths, **kwargs):
    all_x_data = []
    all_y_data = []

    for file_path in file_paths:
        x_survivors = []

        if not kwargs.get('test'):
            X_train = np.load('Neurotech@Rice Datathon Challenge' + file_path + '_NEW_X.npy')
            y_train = np.load('Neurotech@Rice Datathon Challenge' + file_path + '_NEW_y.npy')
            y_train = [int(i) - 1 for i in y_train]
        else:
            X_train = np.load(file_path)
        counter = 0
        if kwargs.get('decimate'):
            # remove every other sample for waking state (y=0)
            for i in range(X_train.shape[0] - 1):
                if (y_train[i] != 0 or counter <= 2):
                    all_y_data.append(y_train[i])
                    x_survivors.append(X_train[i])
                counter += 1
                counter %= 4
            survivors = np.array(x_survivors)
            all_x_data.append(power_matrix(get_EEGs(survivors)))
        else:
            all_x_data.append(power_matrix(get_EEGs(X_train)))
            if not kwargs.get('test'):
                all_y_data.extend(y_train)

    all_x_data = np.concatenate(all_x_data, axis=1)
    if not kwargs.get('test'):
        all_y_data = np.array(all_y_data)
    else:
        all_y_data = None
    return all_x_data.transpose(), all_y_data


def load_files(file_paths, **kwargs):
    # Initialize empty lists to store features and labels
    all_X = []
    all_y = []

    counter = 0
    # Load data from each array and concatenate
    for file_path in file_paths:
        if not kwargs.get('test'):
            X_train = np.load('Neurotech@Rice Datathon Challenge' + file_path + '_NEW_X.npy')
            y_train = np.load('Neurotech@Rice Datathon Challenge' + file_path + '_NEW_y.npy')
            y_train = [int(i) - 1 for i in y_train]
        else:
            X_train = np.load(file_path)
        # # if decimate is true, only use every fifth sample for waking state (y=0)
        if (kwargs.get('decimate')):
            for i in range(len(X_train)):
                if (y_train[i] != 0 or counter == 0):
                    all_X.append(X_train[i].transpose())
                    all_y.append(y_train[i])
                counter += 1
                counter %= 2
        else:
            for i in range(len(X_train)):
                all_X.append(X_train[i].transpose())
                if not kwargs.get('test'):
                    all_y.append(y_train[i])

        #
        # for i in range(len(y_train)):
        #     if(y_train[i] != -1):
        #         all_y.append(y_train[i])
        #         all_X.append(X_train[i])

    # Concatenate all features and labels
    if not kwargs.get('test'):
        all_y = np.array(all_y)
    else:
        all_y = None
    all_X = np.array(all_X)

    return all_X, all_y


def load_xgb_files(file_paths, **kwargs):
    # Initialize empty lists to store features and labels
    all_X = []
    all_y = []
    y_train = []

    counter = 0
    # Load data from each array and concatenate
    for file_path in file_paths:
        if not kwargs.get('test'):
            X_train = np.load('Neurotech@Rice Datathon Challenge' + file_path + '_NEW_X.npy')
            y_train = np.load('Neurotech@Rice Datathon Challenge' + file_path + '_NEW_y.npy')
            y_train = [int(i) - 1 for i in y_train]
        else:
            X_train = np.load(file_path)
        # # if decimate is true, only use every fifth sample for waking state (y=0)
        if kwargs.get('decimate'):
            for i in range(len(X_train)):
                if (y_train[i] != 0 or counter == 0):
                    all_X.append(X_train[i].transpose())
                    for j in range(3000):
                        all_y.append(y_train[i])
                counter += 1
                counter %= 5
        else:
            for i in range(len(X_train)):
                all_X.append(X_train[i].transpose())
                if not kwargs.get('test'):
                    for j in range(3000):
                        all_y.append(y_train[i])
    all_X = np.concatenate(all_X, axis=0)
    if not kwargs.get('test'):
        all_y = np.array(all_y)
    else:
        all_y = None
    return all_X, all_y, y_train


def run_xgb(X_train, y_train, all_eX, all_ey, act_y, **kwargs):
    if kwargs.get('domain') == 'freq':
<<<<<<< Updated upstream
        param = {'objective': 'multi:softmax', 'eta': 0.8,
                 'gamma': 0.3, 'lambda': 1.0, 'alpha': 0.0,
=======
        param = {'objective': 'multi:softmax', 'eta': 0.3,
                 'gamma': 0.0, 'lambda': 1.0, 'alpha': 0.0,
>>>>>>> Stashed changes
                 'max_depth': 6, 'nthread': 8, 'num_class': 6}
    else:
        param = {'objective': 'multi:softmax', 'eta': 0.2, 'max_depth': 6, 'nthread': 8, 'num_class': 6}
    if kwargs.get('mode') == 'load':
        if kwargs.get('domain') == 'freq':
            xgb_model_f = pickle.load(open(xgbf_file_name, "rb"))
            xg_test = xgb.DMatrix(all_eX, label=all_ey)
        else:
            xgb_model = pickle.load(open(xgb_file_name, "rb"))
            xg_test = xgb.DMatrix(all_eX, label=all_ey)
    # else retrain model
    if kwargs.get('mode') == 'train':
        # Assuming all_X is a 2D array with multiple channels
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_test = xgb.DMatrix(all_eX, label=all_ey)
        # xg_train = xgb.DMatrix(np.array(X_train).reshape((1, -1)), label=y_train)
        # xg_test = xgb.DMatrix(np.array(X_test).reshape((1, -1)), label=y_test)

        print("training model")
        # Initialize the XGBoost classifier
        # model = XGBClassifier()

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 5

        # save
        if kwargs.get('freq'):
            xgb_model_f = xgb.train(param, xg_train, num_round, watchlist)
            pickle.dump(xgb_model_f, open(xgbf_file_name, "wb"))
            xgb_model_f.save_model('xgb_freq.model')
            xgb_model_f = xgb.train(param, xg_train, num_round, watchlist)
            pickle.dump(xgb_model_f, open(xgbf_file_name, "wb"))
            xgb_model_f.save_model('xgb_freq.model')
        else:
            xgb_model = xgb.train(param, xg_train, num_round, watchlist)
            pickle.dump(xgb_model, open(xgb_file_name, "wb"))
            xgb_model.save_model('xgb.model')

    if kwargs.get('mode') == 'retrain':
        # Assuming all_X is a 2D array with multiple channels
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_test = xgb.DMatrix(all_eX, label=all_ey)
        # xg_train = xgb.DMatrix(np.array(X_train).reshape((1, -1)), label=y_train)
        # xg_test = xgb.DMatrix(np.array(X_test).reshape((1, -1)), label=y_test)

        print("training model")
        # Initialize the XGBoost classifier
        # model = XGBClassifier()

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 5

        # save
        if kwargs.get('domain') == 'freq':
            xgb_model_f = xgb.train(param, xg_train, num_round, watchlist, xgb_model='xgb_freq.model')
            pickle.dump(xgb_model_f, open(xgbf_file_name, "wb"))
            xgb_model_f.save_model('xgb_freq.model')
            xgb_model_f = xgb.train(param, xg_train, num_round, watchlist)
            pickle.dump(xgb_model_f, open(xgb_file_name, "wb"))
            xgb_model_f.save_model('xgb_freq.model')
        else:
            xgb_model = xgb.train(param, xg_train, num_round, watchlist, xgb_model='xgb.model')
            pickle.dump(xgb_model, open(xgb_file_name, "wb"))
            xgb_model.save_model('xgb.model')

    # get prediction
    if kwargs.get('domain') == 'freq':
        y_pred = xgb_model_f.predict(xg_test)
    else:
        y_pred = xgb_model.predict(xg_test)

    # find most common value in each epoch
    stages = []

    if kwargs.get('domain') == 'freq':
        stages = y_pred
    else:
        for epoch in range(len(act_y)):
            stage_freq = {}
            for i in range(3000):
                if (y_pred[epoch * 3000 + i] in stage_freq):
                    stage_freq[y_pred[epoch * 3000 + i]] += 1
                else:
                    stage_freq[y_pred[epoch * 3000 + i]] = 1
            stages.append(max(stage_freq, key=stage_freq.get))

        stages = np.array(stages)

<<<<<<< Updated upstream
    for i in range(2, len(y_pred) - 2):
        if y_pred[i] != 0 and y_pred[i - 1] == 0 and y_pred[i + 1] == 0:
            y_pred[i] = 0

=======
    for i in range(2, len(stages) - 2):
        if stages[i] != stages[i - 1]:
            stages[i] = stages[i - 1]
        if stages[i] != stages[i + 1]:
            stages[i] = stages[i + 1]

    stages[:3] = np.zeros(3)
    stages[len(stages) - 3:] = np.zeros(3)
>>>>>>> Stashed changes

    # Evaluate the performance of the model
    # Print classification report for detailed metrics
    print("Classification Report For XGB:")
    print(classification_report(act_y, stages))

    # plot feature importance
    # xgb.plot_importance(model)

    # creating the bar plot
    plt.plot(range(len(stages)), stages, color='maroon')
    plt.title('XGB Predicted Stages')
    plt.ylabel('Predicted y')
    plt.xlabel('Epoch')
    plt.show()
    # save plot locally
    plt.savefig('xgb.png')

    plt.plot(range(len(all_ey)), all_ey, color='maroon')

    plt.title('Actual Stages')
    plt.ylabel('Actual y')
    plt.xlabel('Epoch')
    plt.show()
    # save plot locally
    plt.savefig('actual.png')


def train_dnn(X_train, y_train):
    DNN.add(layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))  # Flatten the 2D matrix
    DNN.add(layers.Dense(128, activation='relu'))
    DNN.add(layers.Dense(64, activation='relu'))
    DNN.add(layers.Dense(6, activation='softmax'))  # Output layer with 6 classes

    # Compile the model
    DNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    DNN.fit(X_train, y_train, epochs=10, validation_split=0.2)


def train_cnn(X_train, y_train):
    # Reshape the input matrices for compatibility with Conv2D layer
    X_train = X_train.reshape((-1, X_train.shape[1], X_train.shape[2]))
    # Define the CNN model
    CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dense(6, activation='softmax'))  # Output layer with 6 classes

    # Compile the model
    CNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    CNN.fit(X_train, y_train, epochs=10, validation_split=0.2)


if __name__ == '__main__':
<<<<<<< Updated upstream
    y = call_FRQ('Neurotech@Rice Datathon Challenge/Training/p00_n1_NEW_X.npy')
    plt.plot(range(len(y)), y, color='maroon')
    plt.show()
    # y = call_CNN('Neurotech@Rice Datathon Challenge/Training/p00_n1_NEW_X.npy')
    # plt.plot(range(len(y)), y, color='maroon')
    # plt.show()
    # y = call_DNN('Neurotech@Rice Datathon Challenge/Training/p00_n1_NEW_X.npy')
    # plt.plot(range(len(y)), y, color='maroon')
    # plt.show()

    # X_test, y_act = load_freq_files(test_file_paths, decimate=False, test=False)
    # xgb_model_f = pickle.load(open(xgbf_file_name, "rb"))
    # xg_test = xgb.DMatrix(X_test)
    # y_pred = xgb_model_f.predict(xg_test)
    # # set any non zero values with zeros on either side to zero
    # for i in range(2, len(y_pred) - 2):
    #     if y_pred[i] != 0 and y_pred[i - 1] == 0 and y_pred[i + 1] == 0:
    #         y_pred[i] = 0
    # #save predicted stages to numpy file
    # #np.save('eval_b_y_pred.npy', y_pred)
    # print(classification_report(y_act, y_pred))
    # plt.plot(range(len(y_pred)), y_pred, color='maroon')
    # plt.title('XGB Predicted Stages')
    # plt.show()
    #
    # plt.plot(range(len(y_act)), y_act, color='maroon')
    # plt.title('Actual Stages')
    # plt.show()

    # plot the predicted y values from numpy files
    # y_pred = np.load('eval_a_y_pred.npy')
    # plt.plot(range(len(y_pred)), y_pred, color='maroon')
    # plt.title('XGB Predicted Stages')
    # plt.ylabel('Predicted y')
    # plt.xlabel('Epoch')
    # plt.show()


    # # intake XGB data
    # X_train, y_train, _ = load_xgb_files(file_paths, decimate=True)
    # all_eX, all_ey, act_y = load_xgb_files(test_file_paths, decimate=False)

    X_train = pickle.load(open("X_train.pkl", "rb"))
    y_train = pickle.load(open("y_train.pkl", "rb"))
    all_eX = pickle.load(open("all_eX.pkl", "rb"))
    act_y = pickle.load(open("act_y.pkl", "rb"))

    run_xgb(X_train, y_train, all_eX, act_y, act_y, domain='freq', mode='retrain')

=======
    y = call_FRQ('data/p00_n1_X.npy')
    # plt.plot(range(len(y)), y, color='maroon')
    # plt.show()
    y = call_CNN('data/p00_n1_X.npy')
    plt.plot(range(len(y)), y, color='maroon')
    plt.show()
    y = call_DNN('data/p00_n1_X.npy')
    # plt.plot(range(len(y)), y, color='maroon')
    # plt.show()

    # # # intake XGB data
    # # X_train, y_train, _ = load_xgb_files(file_paths, decimate=True)
    # # all_eX, all_ey, act_y = load_xgb_files(test_file_paths, decimate=False)
    # #
    # # run_xgb(X_train, y_train, all_eX, all_ey, act_y, load=False, train=True, freq=False)
    #
>>>>>>> Stashed changes
    # # # # train the xgb model
    # X_train, y_train = load_freq_files(file_paths, decimate=False)
    # all_eX, act_y = load_freq_files(test_file_paths, decimate=False)
    # # # save above data to pickle file
    # # pickle.dump(X_train, open("X_train.pkl", "wb"))
    # # pickle.dump(y_train, open("y_train.pkl", "wb"))
    # # pickle.dump(all_eX, open("all_eX.pkl", "wb"))
    # # pickle.dump(act_y, open("act_y.pkl", "wb"))
    #
    # #
    # # # # load above data from pickle file
    # # X_train = pickle.load(open("X_train.pkl", "rb"))
    # # y_train = pickle.load(open("y_train.pkl", "rb"))
    # # all_eX = pickle.load(open("all_eX.pkl", "rb"))
    # # act_y = pickle.load(open("act_y.pkl", "rb"))
    # #
    # run_xgb(X_train, y_train, all_eX, act_y, act_y, domain='time', mode='retrain')
    #
    # # file paths for training data
    #
    # # Initialize empty lists to store features and labels
    # X_train, y_train = load_files(file_paths, decimate=True)
    #
    # # Initialize empty lists to store features and labels
    # X_test, y_test = load_files(test_file_paths, decimate=False)
    # X_test = X_test.reshape((-1, X_test.shape[1], X_test.shape[2], 1))
    # # transpose each row of X_test
    # # for i in range(len(X_test)):
    # #     X_test[i] = X_test[i].transpose()
    #
    # # train the deep neural network model
    # train_cnn(X_train, y_train)
    # train_dnn(X_train, y_train)
    #
    # # load the deep neural network model from pickle file
    # # CNN = pickle.load(open(CNN_file_name, "rb"))
    # # DNN = pickle.load(open(DNN_file_name, "rb"))
    #
    # # Evaluate the models on the test set and plot the predicted y values
    # Ctest_loss, Ctest_acc = CNN.evaluate(X_test, y_test)
    # print(f'CNN Test accuracy: {Ctest_acc}')
    #
    # # plot the predicted y values
    #
    # Dtest_loss, Dtest_acc = DNN.evaluate(X_test, y_test)
    # print(f'DNN Test accuracy: {Dtest_acc}')
    #
    # c_y_pred = CNN.predict(X_test)
    # d_y_pred = DNN.predict(X_test)
    #
    # # # find max y value for each epoch, stage is index of max y value
    # c_stage = []
    # d_stage = []
    # for i in range(len(c_y_pred)):
    #     c_stage.append(np.argmax(c_y_pred[i]))
    #
    # for i in range(len(d_y_pred)):
    #     d_stage.append(np.argmax(d_y_pred[i]))
    #
    # # Evaluate the performance of the model
    # # Print classification report for detailed metrics
    # print("Classification Report for CNN:")
    # print(classification_report(y_test, c_stage))
    # print("Classification Report for DNN:")
    # print(classification_report(y_test, d_stage))
    # # creating the bar plot
    # plt.plot(range(len(c_stage)), c_stage, color='maroon')
    #
    # plt.ylabel('Predicted y')
    # plt.xlabel('Epoch')
    # plt.show()
    #
    # # save the models to pickle files
    # pickle.dump(CNN, open(CNN_file_name, "wb"))
    # pickle.dump(DNN, open(DNN_file_name, "wb"))
