import datetime
import pickle
import threading
import time

import heartpy as hp
import numpy as np

from pythonosc import dispatcher, osc_server

from FeatureExtractionEDA import process_eda
from FeatureExtractionPPG import process_ppg
from FeatureExtractionTEMP import process_temp
from paths_config import Paths

models_path = Paths.models_path
with open(models_path + '/RF_model_valence.pkl', 'rb') as f:
    valence_model = pickle.load(f)

with open(models_path + '/RF_model_arousal.pkl', 'rb') as f:
    arousal_model = pickle.load(f)

print('Models loaded')


def compute_sampling_frequency(tmsp):
    # Define filter parameters
    timer = []

    for i in range(len(tmsp)):
        timer.append(datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(tmsp[i]), '%Y-%m-%d %H:%M:%S.%f'))

    sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')

    return sample_rate


def feature_extraction(ppg, tmsp_ppg, eda, tmsp_eda, temp, tmsp_temp):
    feats_ppg = process_ppg(ppg, tmsp_ppg)
    feats_eda = process_eda(eda, tmsp_eda)
    feats_temp = process_temp(temp, tmsp_temp)

    features = np.hstack((np.array([float(value) for value in feats_ppg.values()]),
                          np.array([float(value) for value in feats_eda.values()]),
                          np.array([float(value) for value in feats_temp.values()])))

    features[np.isinf(features)] = 0
    features[np.isnan(features)] = 0

    return features


def emotion_prediction(ppg, tmsp_ppg, eda, tmsp_eda, temp, tmsp_temp):
    features = feature_extraction(ppg, tmsp_ppg, eda, tmsp_eda, temp, tmsp_temp)

    features = features.reshape(1, -1)

    valence = valence_model.predict(features)
    arousal = arousal_model.predict(features)

    return valence, arousal


class EmotiBitAnalysis:

    def __init__(self):
        self.osc_thread = None
        self.server = None
        self.dispatcher = None
        self.valence = None
        self.arousal = None
        self.ppg = []
        self.tmsp_ppg = []
        self.eda = []
        self.tmsp_eda = []
        self.temp = []
        self.tmsp_temp = []

        self.emotion_recognition()

    def check_signals(self):
        # Compute sampling frequencies and length of windows
        fs_ppg = compute_sampling_frequency(self.tmsp_ppg)
        window_ppg = int(30 * fs_ppg)
        fs_eda = compute_sampling_frequency(self.tmsp_eda)
        window_eda = int(30 * fs_eda)
        fs_temp = compute_sampling_frequency(self.tmsp_temp)
        window_temp = int(30 * fs_temp)

        # Check if there are 30 seconds of data
        if len(self.ppg) >= window_ppg and len(self.eda) >= window_eda and len(self.temp) >= window_temp:
            # Estimate valence and arousal
            self.valence, self.arousal = emotion_prediction(self.ppg, self.tmsp_ppg,
                                                            self.eda, self.tmsp_eda,
                                                            self.temp, self.tmsp_temp)

            time.sleep(5)

            # Handle results
            print('Valence: ', self.valence[0])
            print('Arousal: ', self.arousal[0])

            # Pop 15 seconds
            self.ppg = self.ppg[int(15 * fs_ppg):]
            self.tmsp_ppg = self.tmsp_ppg[int(15 * fs_ppg):]
            self.eda = self.eda[int(15 * fs_eda):]
            self.tmsp_eda = self.tmsp_eda[int(15 * fs_eda):]
            self.temp = self.temp[int(15 * fs_temp):]
            self.tmsp_temp = self.tmsp_temp[int(15 * fs_temp):]

    # Function to handle incoming OSC messages
    def osc_handler(self, address, *args):
        channel_name = address[12:len(address)]
        if channel_name == 'PPG:GRN':
            self.ppg.append(args)
            dt = datetime.datetime.now(datetime.timezone.utc)

            utc_time = dt.replace(tzinfo=datetime.timezone.utc)
            utc_timestamp = utc_time.timestamp()
            self.tmsp_ppg.append(utc_timestamp)
        elif channel_name == 'EDA':
            self.eda.append(args)
            dt = datetime.datetime.now(datetime.timezone.utc)

            utc_time = dt.replace(tzinfo=datetime.timezone.utc)
            utc_timestamp = utc_time.timestamp()
            self.tmsp_eda.append(utc_timestamp)
        elif channel_name == 'THERM':
            self.temp.append(args)
            dt = datetime.datetime.now(datetime.timezone.utc)

            utc_time = dt.replace(tzinfo=datetime.timezone.utc)
            utc_timestamp = utc_time.timestamp()
            self.tmsp_temp.append(utc_timestamp)

        try:
            self.check_signals()
        except:
            pass

    def start_server(self):
        # Create an OSC dispatcher
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/EmotiBit/*", self.osc_handler)

        # Start an OSC server
        self.server = osc_server.ThreadingOSCUDPServer(("localhost", 12345), self.dispatcher)
        print("OSC server started")

        self.server.serve_forever()

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()

    def emotion_recognition(self):
        self.osc_thread = threading.Thread(target=self.start_server)
        if not self.osc_thread.is_alive():
            self.osc_thread.start()


def main():
    EmotiBitAnalysis()


if __name__ == "__main__":
    main()
