from dataprocessing import filters
from dataprocessing import dataprep
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, GlobalMaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import joblib


if __name__ == '__main__':

    train_map = {
        1: 1.2 * np.array([0, 0]) + 0.3 * np.array([1, 0]),
        2: 1.2 * np.array([1, 0]) + 0.3 * np.array([1, 0]),
        3: 1.2 * np.array([2, 0]) + 0.3 * np.array([1, 0]),
        4: 1.2 * np.array([3, 0]) + 0.3 * np.array([1, 0]),
        5: 1.2 * np.array([4, 0]) + 0.3 * np.array([1, 0]),
        6: 1.2 * np.array([4, 2]) + 0.3 * np.array([1, 0]),
        7: 1.2 * np.array([3, 2]) + 0.3 * np.array([1, 0]),
        8: 1.2 * np.array([2, 2]) + 0.3 * np.array([1, 0]),
        9: 1.2 * np.array([1, 2]) + 0.3 * np.array([1, 0]),
        10: 1.2 * np.array([0, 2]) + 0.3 * np.array([1, 0])
    }

    test_map = {
        1: 1.2 * np.array([0, 0]) + 0.3 * np.array([0, 0]),
        2: 1.2 * np.array([1, 0]) + 0.3 * np.array([0, 0]),
        3: 1.2 * np.array([2, 0]) + 0.3 * np.array([0, 0]),
        4: 1.2 * np.array([3, 0]) + 0.3 * np.array([0, 0]),
        5: 1.2 * np.array([4, 0]) + 0.3 * np.array([0, 0]),
        6: 1.2 * np.array([4, 2]) + 0.3 * np.array([0, 0]),
        7: 1.2 * np.array([3, 2]) + 0.3 * np.array([0, 0]),
        8: 1.2 * np.array([2, 2]) + 0.3 * np.array([0, 0]),
        9: 1.2 * np.array([1, 2]) + 0.3 * np.array([0, 0]),
        10: 1.2 * np.array([0, 2]) + 0.3 * np.array([0, 0])
    }

    loc_num = 10
    loc_list = np.arange(1, loc_num + 1)
    loc_list_num = len(loc_list)
    f_train_path = "dataset/tc204/train/"

    # create raw train data map
    rtrain_map_csi_a = {}
    rtrain_map_csi_b = {}
    rtrain_map_csi_c = {}
    for loc in loc_list:
        f_name = f_train_path + "train_" + str(loc) + ".save"
        data_list = joblib.load(f_name)
        tmp_csi_a_list = []
        tmp_csi_b_list = []
        tmp_csi_c_list = []
        for data in data_list:
            ntx = data["Ntx"]
            nrx = data["Nrx"]
            tmp_csi_list = dataprep.get_scaled_csi(data)
            for m in range(ntx):
                for n in range(nrx):
                    tmp_csi = tmp_csi_list[:, m, n]
                    if n == 0:
                        tmp_csi_a_list.append(tmp_csi)
                    if n == 1:
                        tmp_csi_b_list.append(tmp_csi)
                    if n == 2:
                        tmp_csi_c_list.append(tmp_csi)

        rtrain_map_csi_a[loc] = np.vstack(tmp_csi_a_list)
        rtrain_map_csi_b[loc] = np.vstack(tmp_csi_b_list)
        rtrain_map_csi_c[loc] = np.vstack(tmp_csi_c_list)

    # create filtered train data map
    ftrain_map_csi_a = {}
    ftrain_map_csi_b = {}
    ftrain_map_csi_c = {}
    filter_window = 10
    center_freq = 2.4e9

    for loc in loc_list:
        tmp_csi_a_list = rtrain_map_csi_a[loc]
        tmp_csi_b_list = rtrain_map_csi_b[loc]
        tmp_csi_c_list = rtrain_map_csi_c[loc]

        f_csi_a_list = filters.cir_filter(np.vstack(tmp_csi_a_list), center_freq, filter_window)
        f_csi_b_list = filters.cir_filter(np.vstack(tmp_csi_b_list), center_freq, filter_window)
        f_csi_c_list = filters.cir_filter(np.vstack(tmp_csi_c_list), center_freq, filter_window)

        num_samples = f_csi_a_list.shape[0]
        num_packets = 3
        num_out = num_samples - num_packets + 1

        tmp_csi_a = []
        tmp_csi_b = []
        tmp_csi_c = []

        for i in range(num_out):
            pkt = np.zeros((num_packets, 30), dtype='complex')
            pkt_max = np.zeros((num_packets, ))
            for k in range(num_packets):
                pkt[k, :] = f_csi_a_list[i + k, :]
                pkt_max[k] = np.mean(np.abs(pkt[k, :]))

            pkt_max_id = np.argmax(pkt_max)
            tmp_csi_a.append(pkt[pkt_max_id, :])

        for i in range(num_out):
            pkt = np.zeros((num_packets, 30), dtype='complex')
            pkt_max = np.zeros((num_packets,))
            for k in range(num_packets):
                pkt[k, :] = f_csi_b_list[i + k, :]
                pkt_max[k] = np.mean(np.abs(pkt[k, :]))

            pkt_max_id = np.argmax(pkt_max)
            tmp_csi_b.append(pkt[pkt_max_id, :])

        for i in range(num_out):
            pkt = np.zeros((num_packets, 30), dtype='complex')
            pkt_max = np.zeros((num_packets,))
            for k in range(num_packets):
                pkt[k, :] = f_csi_c_list[i + k, :]
                pkt_max[k] = np.mean(np.abs(pkt[k, :]))

            pkt_max_id = np.argmax(pkt_max)
            tmp_csi_c.append(pkt[pkt_max_id, :])

        ftrain_map_csi_a[loc] = np.vstack(tmp_csi_a)
        ftrain_map_csi_b[loc] = np.vstack(tmp_csi_b)
        ftrain_map_csi_c[loc] = np.vstack(tmp_csi_c)

    # generate testing data set
    f_test_path = "dataset/tc204/test/"

    # create raw train data map
    rtest_map_csi_a = {}
    rtest_map_csi_b = {}
    rtest_map_csi_c = {}
    for loc in loc_list:
        f_name = f_test_path + "test_" + str(loc) + ".save"
        data_list = joblib.load(f_name)
        tmp_csi_a_list = []
        tmp_csi_b_list = []
        tmp_csi_c_list = []
        for data in data_list:
            ntx = data["Ntx"]
            nrx = data["Nrx"]
            tmp_csi_list = dataprep.get_scaled_csi(data)
            for m in range(ntx):
                for n in range(nrx):
                    tmp_csi = tmp_csi_list[:, m, n]
                    if n == 0:
                        tmp_csi_a_list.append(tmp_csi)
                    if n == 1:
                        tmp_csi_b_list.append(tmp_csi)
                    if n == 2:
                        tmp_csi_c_list.append(tmp_csi)

        rtest_map_csi_a[loc] = np.vstack(tmp_csi_a_list)
        rtest_map_csi_b[loc] = np.vstack(tmp_csi_b_list)
        rtest_map_csi_c[loc] = np.vstack(tmp_csi_c_list)

    # create filtered test data map
    ftest_map_csi_a = {}
    ftest_map_csi_b = {}
    ftest_map_csi_c = {}

    for loc in loc_list:
        tmp_csi_a_list = rtest_map_csi_a[loc]
        tmp_csi_b_list = rtest_map_csi_b[loc]
        tmp_csi_c_list = rtest_map_csi_c[loc]

        f_csi_a_list = filters.cir_filter(tmp_csi_a_list, center_freq, filter_window)
        f_csi_b_list = filters.cir_filter(tmp_csi_b_list, center_freq, filter_window)
        f_csi_c_list = filters.cir_filter(tmp_csi_c_list, center_freq, filter_window)

        num_samples = f_csi_a_list.shape[0]
        num_packets = 3
        num_out = num_samples - num_packets + 1

        tmp_csi_a = []
        tmp_csi_b = []
        tmp_csi_c = []

        for i in range(num_out):
            pkt = np.zeros((num_packets, 30), dtype='complex')
            pkt_max = np.zeros((num_packets,))
            for k in range(num_packets):
                pkt[k, :] = f_csi_a_list[i + k, :]
                pkt_max[k] = np.mean(np.abs(pkt[k, :]))

            pkt_max_id = np.argmax(pkt_max)
            tmp_csi_a.append(pkt[pkt_max_id, :])

        for i in range(num_out):
            pkt = np.zeros((num_packets, 30), dtype='complex')
            pkt_max = np.zeros((num_packets,))
            for k in range(num_packets):
                pkt[k, :] = f_csi_b_list[i + k, :]
                pkt_max[k] = np.mean(np.abs(pkt[k, :]))

            pkt_max_id = np.argmax(pkt_max)
            tmp_csi_b.append(pkt[pkt_max_id, :])

        for i in range(num_out):
            pkt = np.zeros((num_packets, 30), dtype='complex')
            pkt_max = np.zeros((num_packets,))
            for k in range(num_packets):
                pkt[k, :] = f_csi_c_list[i + k, :]
                pkt_max[k] = np.mean(np.abs(pkt[k, :]))

            pkt_max_id = np.argmax(pkt_max)
            tmp_csi_c.append(pkt[pkt_max_id, :])

        ftest_map_csi_a[loc] = np.vstack(tmp_csi_a)
        ftest_map_csi_b[loc] = np.vstack(tmp_csi_b)
        ftest_map_csi_c[loc] = np.vstack(tmp_csi_c)

    # --------------------------------------------------------------------------------
    # create train dataset
    # --------------------------------------------------------------------------------
    # packet_num = 960
    # packet_offset = 10
    # packet_per_image = 5

    train_image_map = {}

    for loc in loc_list:
        tmp_csi_a_list = ftrain_map_csi_a[loc]
        tmp_csi_b_list = ftrain_map_csi_b[loc]
        tmp_csi_c_list = ftrain_map_csi_c[loc]

        csi_a_mag = np.abs(tmp_csi_a_list)
        csi_b_mag = np.abs(tmp_csi_b_list)
        csi_c_mag = np.abs(tmp_csi_c_list)

        csi_a_phase = np.unwrap(np.angle(tmp_csi_a_list))
        csi_b_phase = np.unwrap(np.angle(tmp_csi_b_list))
        csi_c_phase = np.unwrap(np.angle(tmp_csi_c_list))

        # phase calibration
        k = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])

        aa = (csi_a_phase[:, -1] - csi_a_phase[:, 0]) / (k[-1] - k[0])
        ba = (1 / 30) * np.sum(csi_a_phase, axis=1)

        ab = (csi_b_phase[:, -1] - csi_b_phase[:, 0]) / (k[-1] - k[0])
        bb = (1 / 30) * np.sum(csi_b_phase, axis=1)

        ac = (csi_c_phase[:, -1] - csi_c_phase[:, 0]) / (k[-1] - k[0])
        bc = (1 / 30) * np.sum(csi_c_phase, axis=1)

        clean_phase_a = np.zeros((csi_a_phase.shape[0], 30))
        clean_phase_b = np.zeros((csi_b_phase.shape[0], 30))
        clean_phase_c = np.zeros((csi_c_phase.shape[0], 30))

        for i in range(30):
            clean_phase_a[:, i] = csi_a_phase[:, i] - (aa * k[i]) - ba
            clean_phase_b[:, i] = csi_b_phase[:, i] - (ab * k[i]) - bb
            clean_phase_c[:, i] = csi_c_phase[:, i] - (ac * k[i]) - bc

        # create images
        image_num = tmp_csi_a_list.shape[0]
        tmp_image = np.zeros((image_num, 30, 6))

        for i in range(image_num):
            tmp_image[i, :, :] = np.vstack((csi_a_mag[i, :], csi_b_mag[i, :],  csi_c_mag[i, :], csi_a_phase[i, :],
                                            csi_b_phase[i, :], csi_c_phase[i, :])).T
        train_image_map[loc] = tmp_image

    xtrain_data_list = []
    ytrain_data_list = []

    for loc in loc_list:
        tmp_x = train_image_map[loc]
        xtrain_data_list.append(tmp_x)
        ytrain_data_list.append(np.tile([loc], tmp_x.shape[0]))

    xtrain_dataset = np.vstack(xtrain_data_list)
    ytrain_dataset = np.hstack(ytrain_data_list)

    # normalize data
    num_row = xtrain_dataset.shape[0]

    x_ = np.vstack(xtrain_dataset)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_)
    x_ = x_scaler.transform(x_)
    xtrain_dataset = np.reshape(x_, (num_row, 30, 6, 1))

    # create one-hot encoding
    yencoder = LabelEncoder()
    yencoder.fit(ytrain_dataset)
    ytrain_dataset = yencoder.transform(ytrain_dataset)
    ytrain_dataset = np_utils.to_categorical(ytrain_dataset)

    # --------------------------------------------------------------------------------
    # create test dataset
    # --------------------------------------------------------------------------------
    test_image_map = {}

    for loc in loc_list:
        tmp_csi_a_list = ftest_map_csi_a[loc]
        tmp_csi_b_list = ftest_map_csi_b[loc]
        tmp_csi_c_list = ftest_map_csi_c[loc]

        csi_a_mag = np.abs(tmp_csi_a_list)
        csi_b_mag = np.abs(tmp_csi_b_list)
        csi_c_mag = np.abs(tmp_csi_c_list)

        csi_a_phase = np.unwrap(np.angle(tmp_csi_a_list))
        csi_b_phase = np.unwrap(np.angle(tmp_csi_b_list))
        csi_c_phase = np.unwrap(np.angle(tmp_csi_c_list))

        # phase calibration
        k = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])

        aa = (csi_a_phase[:, -1] - csi_a_phase[:, 0]) / (k[-1] - k[0])
        ba = (1 / 30) * np.sum(csi_a_phase, axis=1)

        ab = (csi_b_phase[:, -1] - csi_b_phase[:, 0]) / (k[-1] - k[0])
        bb = (1 / 30) * np.sum(csi_b_phase, axis=1)

        ac = (csi_c_phase[:, -1] - csi_c_phase[:, 0]) / (k[-1] - k[0])
        bc = (1 / 30) * np.sum(csi_c_phase, axis=1)

        clean_phase_a = np.zeros((csi_a_phase.shape[0], 30))
        clean_phase_b = np.zeros((csi_b_phase.shape[0], 30))
        clean_phase_c = np.zeros((csi_c_phase.shape[0], 30))

        for i in range(30):
            clean_phase_a[:, i] = csi_a_phase[:, i] - (aa * k[i]) - ba
            clean_phase_b[:, i] = csi_b_phase[:, i] - (ab * k[i]) - bb
            clean_phase_c[:, i] = csi_c_phase[:, i] - (ac * k[i]) - bc

        # create images
        image_num = tmp_csi_a_list.shape[0]
        tmp_image = np.zeros((image_num, 30, 6))

        for i in range(image_num):
            tmp_image[i, :, :] = np.vstack((csi_a_mag[i, :], csi_b_mag[i, :], csi_c_mag[i, :], csi_a_phase[i, :],
                                            csi_b_phase[i, :], csi_c_phase[i, :])).T

        test_image_map[loc] = tmp_image

    xtest_data_list = []
    ytest_data_list = []

    for loc in loc_list:
        tmp_x = test_image_map[loc]
        xtest_data_list.append(tmp_x)
        ytest_data_list.append(np.tile([loc], tmp_x.shape[0]))

    xtest_dataset = np.vstack(xtest_data_list)
    ytest_dataset = np.hstack(ytest_data_list)

    # normalize data
    num_row = xtest_dataset.shape[0]

    x_ = np.vstack(xtest_dataset)
    x_ = x_scaler.transform(x_)
    xtest_dataset = np.reshape(x_, (num_row, 30, 6, 1))

    # separate data into train / validation / test set
    x_train, x_val, y_train, y_val = train_test_split(xtrain_dataset, ytrain_dataset, test_size=0.5, shuffle=True,
                                                      random_state=7)

    # create model
    f_save_path = 'results/'
    for count in range(3):
        model = Sequential()
        model.add(Conv2D(20, (13, 6), input_shape=(30, 6, 1), activation='relu'))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10000, verbose=0, callbacks=[es])
        scores = model.evaluate(x_train, y_train, verbose=0)
        print("count = {}, model training accuracy = {:.2f}%".format(count, scores[1] * 100))

        # print(model.summary())
        model.save(f_save_path + "model_2dcnn" + "_c" + str(count) + ".h5")
        joblib.dump(x_scaler, f_save_path + "scaler_2dcnn_" + "_c" + str(count) + ".save")
        # print("model saved!")

        y_pred = model.predict(xtest_dataset)
        num_samples = y_pred.shape[0]
        tru_pos = []
        est_pos = []
        for i in range(num_samples):
            weights = y_pred[i, :]
            tmp_y = np.array([0, 0])
            for k in range(len(weights)):
                tmp_y = tmp_y + (weights[k] * train_map[loc_list[k]])

            est_pos.append(tmp_y)
            tru_pos.append(test_map[ytest_dataset[i]])

        np_est_pos = np.vstack(est_pos)
        np_tru_pos = np.vstack(tru_pos)

        pos_err = np.sqrt(np.sum(np.square(np_est_pos - np_tru_pos), axis=1))
        pos_err_rmse = np.sqrt(np.mean(np.sum(np.square(np_est_pos - np_tru_pos), axis=1)))
        err50pc = np.percentile(pos_err, 50, interpolation='linear')
        err80pc = np.percentile(pos_err, 80, interpolation='linear')
        print("rmse = {:.2f}, median = {:.2f}, 80th% = {:.2f}".format(pos_err_rmse, err50pc, err80pc))

        np_hist, np_bins = np.histogram(pos_err, bins=np.arange(0, 8, 0.1), density=True)
        dx = np_bins[1] - np_bins[0]
        cdf = np.cumsum(np_hist) * dx
        cdf_x = np_bins[1:]
        cdf_y = cdf
        plt.figure()
        plt.plot(cdf_x, cdf_y)
        plt.grid(True)
        plt.savefig(f_save_path + "fig_cdf_2dcnn" + "_c" + str(count) + ".png")
        # plt.show()
        joblib.dump(cdf_x, "cdf_x_2dcnn.save")
        joblib.dump(cdf_y, "cdf_y_2dcnn.save")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'validate'])
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.savefig(f_save_path + "fig_train_2dcnn" + "_c" + str(count) + ".png")

        plt.figure()
        plt.hist(pos_err, color="black", bins=np.arange(10), density=True)
        plt.grid(True)
        plt.box(True)
        plt.savefig(f_save_path + "fig_2dcnn" + "_c" + str(count) + ".png")
        # plt.show()
