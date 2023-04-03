# To Ease-up nuitka --follow imports in a clean-way, I'm using following imports altogether:
import os, sys, glob, time, winsound, datetime, warnings, PyQt5, numpy, scipy, keras, joblib, h5py, tensorflow, tensorflow.compat.v2, sklearn.preprocessing, plotly, pandas, keras_pos_embd, cloverleaf.GUI_BG, cloverleaf.GUI, cloverleaf.Predict
#####################################################################################

import os, glob, time
import numpy as np
import scipy
import pandas as pd
from scipy.io import loadmat   # this is the SciPy module that loads mat-files
import winsound   # When process finished, play a notification sound
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import datetime
import plotly
import plotly.graph_objects as go


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Dropout, Concatenate, Add
from keras_pos_embd import PositionEmbedding
import tensorflow.compat.v2        # To avoid Nuitka compilation error
from keras import __version__        # To avoid Nuitka compilation error

import sys
# from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


import cloverleaf.GUI as GUI
import cloverleaf.Predict as Predict

import warnings
warnings.filterwarnings("ignore")

# Ignore Tensorflow Warnings
'''
In detail:-
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class HomePage(QDialog):
    def __init__(self):
        super(HomePage, self).__init__()
        # loadUi("GUI.ui", self)      # loadUi("Predict.ui", self)       # When using loadUi instead of following two lines, use only self instead of self.ui
        self.ui = GUI.Ui_Dialog()
        self.ui.setupUi(self)

        self.timer = QTimer()
        self.timer.setSingleShot(True)   # Run Timer only once, not repeatedly.
        self.timer.start(8000)           # 8000 milli-seconds
        self.timer.timeout.connect(self.gotoPrediction)

    def gotoPrediction(self):
        widget.addWidget(PredictScreen())

        # Make Window-with-Frame again
        flags = Qt.WindowFlags(Qt.WindowStaysOnTopHint)
        widget.setWindowFlags(flags)
        widget.showMaximized()

        widget.setCurrentIndex(widget.currentIndex() + 1)

class PredictScreen(QDialog):
    def __init__(self):
        super(PredictScreen, self).__init__()

        # loadUi("Predict.ui", self)       # When using loadUi instead of following two lines, use only self instead of self.ui
        self.ui = Predict.Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.BrowseButton.setHidden(True)
        self.ui.BrowseText.setHidden(True)
        self.ui.progressBar.setHidden(True)
        self.ui.graphicsView.setHidden(True)
        self.ui.dial.setHidden(True)
        self.ui.dial_label.setHidden(True)
        self.ui.webEngineView.setHidden(True)

        self.ui.radioEntireFolder.clicked.connect(self.radio_selection)
        self.ui.radioParticularTrial.clicked.connect(self.radio_selection)

    def radio_selection(self):
        # Fetch Current date-time & make global variable
        global date_time
        date_time = datetime.datetime.now().strftime('_%a_%b_%d_%Y_%Ihr%Mmin%Ssec')
        self.ui.progressBar.setHidden(True)
        self.ui.graphicsView.setHidden(True)
        self.ui.dial.setHidden(True)
        self.ui.dial_label.setHidden(True)
        self.ui.webEngineView.setHidden(True)
        self.ui.BrowseButton.setEnabled(True)
        self.ui.BrowseButton.setHidden(False)
        self.ui.BrowseText.setHidden(False)
        self.ui.BrowseText.setText('')

        if self.ui.radioEntireFolder.isChecked():
            self.ui.BrowseButton.setText('Browse Data Folder')
            self.ui.BrowseButton.clicked.connect(self.browsefiles)

        elif self.ui.radioParticularTrial.isChecked():
            self.ui.BrowseButton.setText('Browse *.mat File')
            self.ui.BrowseButton.clicked.connect(self.browsefiles)

    def browsefiles(self):
        if self.ui.radioEntireFolder.isChecked():
            folder = QFileDialog.getExistingDirectory(self, caption='Choose Data Directory Containing (*.mat) Files', directory=os.getcwd())
            if not folder:
                self.reset_radio_buttons()
                return
            if os.path.isdir(folder):
                self.ui.BrowseText.setText(folder)
                self.all_file_dir_list = sorted(glob.glob(os.path.join(folder, '*.mat')))
                self.all_file_name_list = [x.replace('.mat', "") for x in [x.replace('\\', "") for x in [x.replace(folder, "") for x in self.all_file_dir_list]]]
                self.reset_radio_buttons()
                self.process_start()

        elif self.ui.radioParticularTrial.isChecked():
            all_file_dir_list = QFileDialog.getOpenFileName(self, 'Select Particular Trial', os.getcwd(), 'MAT-File (*.mat)')[0]
            if not all_file_dir_list:
                self.reset_radio_buttons()
                return
            if os.path.isfile(all_file_dir_list):
                self.ui.BrowseText.setText(all_file_dir_list)
                self.all_file_dir_list = [all_file_dir_list]
                _, all_file_name_list = os.path.split(self.all_file_dir_list[0])             # head, tail = os.path.split(url)
                self.all_file_name_list = [all_file_name_list.replace('.mat', "")]
                self.reset_radio_buttons()
                self.process_start()

    def reset_radio_buttons(self):
        # Reset Radio Buttons
        for radio_button in [self.ui.radioEntireFolder, self.ui.radioParticularTrial]:
            radio_button.setAutoExclusive(False)
            radio_button.setChecked(False)
            radio_button.repaint()
            radio_button.setAutoExclusive(True)
        return


    def process_start(self):
        self.ui.BrowseButton.setEnabled(False)
        self.ui.progressBar.setHidden(False)
        self.ui.progressBar.setValue(0)    # setting value to progress bar
        # When process starts, play a notification sound
        winsound.PlaySound('.\\cloverleaf\\processstart.wav', winsound.SND_FILENAME)

        self.worker = background_dnn_processing_Thread(self.all_file_dir_list, self.all_file_name_list)
        self.worker.start()
        self.worker.update_BrowseButtonText.connect(self.process_update_BrowseButtonText)
        self.worker.update_progressBar.connect(self.process_update_ProgressBar)
        self.worker.update_BrowseText.connect(self.process_finished_update_BrowseText)
        self.worker.finished.connect(self.process_finished)

    def process_update_BrowseButtonText(self, str):
        self.ui.BrowseButton.setText(str)

    def process_update_ProgressBar(self, val):
        self.ui.progressBar.setValue(val)

    def process_finished_update_BrowseText(self, str):
        self.ui.BrowseButton.setText('Classification Saved')
        self.ui.BrowseText.setText(str)
        self.saved_path = str

    def process_finished(self):
        # When process finished, play a notification sound
        winsound.PlaySound('.\\cloverleaf\\processed.wav', winsound.SND_FILENAME)

        self.delay_worker = process_finished_delay_Thread()
        self.delay_worker.start()
        self.delay_worker.finished.connect(self.plot_prediction)


    def plot_prediction(self):
        self.ui.graphicsView.setHidden(False)
        self.ui.dial.setHidden(False)
        self.ui.dial_label.setHidden(False)

        self.ui.dial.setRange(1, len(self.all_file_name_list))
        self.ui.dial.setNotchTarget(1.0)

        self.ui.webEngineView.setHidden(False)
        self.ui.dial_label.setText('{}/{}'.format(1, len(self.all_file_name_list)))
        # create the plotly figure
        # we create an instance of QWebEngineView and set the html code
        html = self.create_plotly_html(1)
        self.ui.webEngineView.setHtml(html)

        self.worker1 = background_Timer_Thread()
        self.worker1.start()
        self.worker1.update_Title_2_Text.connect(self.update_Title_2)

        self.ui.dial.valueChanged.connect(self.dial_method)


    def update_Title_2(self, str):
        self.ui.Title_2.setText(str)

    def dial_method(self):
        value = int(self.ui.dial.value())
        self.ui.dial_label.setText('{}/{}'.format(value, len(self.all_file_name_list)))

        # create the plotly figure
        html = self.create_plotly_html(value-1)
        # we create an instance of QWebEngineView and set the html code
        self.ui.webEngineView.setHtml(html)

    def create_plotly_html(self, dial_value):
        fig = go.Figure()

        pred_file_path = '.\\classified_labels{}\\Y_pred\\{}.csv'.format(date_time, self.all_file_name_list[dial_value - 1])
        Y_pred_plot = pd.read_csv(pred_file_path)

        x_index = [(i + 1) for i in range(0, len(Y_pred_plot.iloc[:, 0]))]

        if os.path.exists('.\\classified_labels{}\\Y_true'.format(date_time)):
            true_file_path = '.\\classified_labels{}\\Y_true\\{}.csv'.format(date_time, self.all_file_name_list[dial_value - 1])
            Y_true_plot = pd.read_csv(true_file_path)
            fig.add_trace(go.Scatter(x = x_index, y = Y_true_plot.iloc[:, 0], name='True',
                                     line=dict(color='green', width=4.5, dash='dashdot')))

        fig.add_trace(go.Scatter(x = x_index, y = Y_pred_plot.iloc[:, 0], name='Prediction',
                                 line=dict(color='blue', width=4.5)))

        # Edit the layout
        title = {'text': 'Classification Plot (File - {})'.format(self.all_file_name_list[dial_value - 1]), 'font': dict(size=30),
                 'y': 0.97, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}

        legend = dict(orientation="h", yanchor="bottom", y=1.002, xanchor="center", x=0.913, bgcolor=None,
                      bordercolor=None, borderwidth=0)  # bgcolor="LightSteelBlue"

        fig.update_layout(font_family="Bookman Old Style", font_color="black",
                          title_font_family="Bookman Old Style", title_font_color="black", font_size = 18,
                          legend_title_font_color="black")

        fig.update_layout(hovermode='x')
        fig.update_layout(title=title, legend=legend, xaxis_title='WiFi Packets',
                          yaxis_title='Human-to-Human Interaction', width=1556, height=615)
        fig.update_layout(paper_bgcolor = '#A5C8FF')   # , plot_bgcolor=<VALUE>

        ########################################################################
        # we create html code of the figure
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        return html


class background_dnn_processing_Thread(QThread):
    def __init__(self, all_file_dir_list, all_file_name_list):
        super(background_dnn_processing_Thread, self).__init__()
        self.all_file_dir_list = all_file_dir_list
        self.all_file_name_list = all_file_name_list

    # Define Signal
    update_BrowseButtonText = pyqtSignal(str)
    update_progressBar = pyqtSignal(int)
    update_BrowseText = pyqtSignal(str)

    def run(self):
        progressBar_count, total_file = 0, 2*len(self.all_file_dir_list)      # 2 times bcz we'll update progress bar twice for single file

        for mat_file, file_name in zip(self.all_file_dir_list, self.all_file_name_list):
            self.update_BrowseButtonText.emit('Preprocessing File-{}'.format(int(progressBar_count/2+1)))
            input, Target_Label, Target_Label_Flag = data_processing_classification(mat_file, file_name).xy_split()
            # Target_Label_Flag = True --> means Dataset has Target label, You can plot True Trace in prediction plot
            progressBar_count += 1
            self.update_progressBar.emit(int(progressBar_count * 100 / total_file))

            self.update_BrowseButtonText.emit('Classifying File-{}'.format(int((progressBar_count+1)/2)))
            saved_path = data_processing_classification(mat_file, file_name).classify(input, Target_Label, Target_Label_Flag)
            progressBar_count += 1
            self.update_progressBar.emit(int(progressBar_count*100/total_file))

        self.update_BrowseText.emit(saved_path)



class data_processing_classification():
    def __init__(self, mat_file, file_name):
        super(data_processing_classification, self).__init__()
        self.mat_file = mat_file
        self.file_name = file_name
        self.min_packets = 1560
        self.mode_point = 20
        self.sliding_points = 1
        self.cols = list(np.load('.\\cloverleaf\\cnpy.npy'))
        # Robust Scaler Scales features using statistics that are robust to outliers
        self.sc = joblib.load('.\\cloverleaf\\frspy.joblib')  # load the Robust Scaler model
        self.enc = joblib.load('.\\cloverleaf\\tohpy.joblib')  # Load Target One Hot Encoded Object
        self.label_dict = {
            'I1': 'Approaching', 'I2': 'Departing', 'I3': 'Handshaking', 'I4': 'High five', 'I5': 'Hugging',
            'I6': 'Kicking (left leg)', 'I7': 'Kicking (right leg)', 'I8': 'Pointing (left hand)',
            'I9': 'Pointing (right hand)', 'I10': 'Punching (left hand)', 'I11': 'Punching (right hand)',
            'I12': 'Pushing', 'I13': 'Steady state'
        }

        # Load Models
        self.number_of_models = len(sorted(glob.glob(os.path.join(os.getcwd(), 'cloverleaf', '*.h5'))))
        self.model = []
        for model_path in sorted(glob.glob(os.path.join(os.getcwd(), 'cloverleaf', '*.h5'))):
            self.model.append(load_model(model_path, custom_objects = {"PositionEmbedding": PositionEmbedding}))


    def xy_split(self):
        min_packets, cols, mat_file = self.min_packets, self.cols, self.mat_file
        Target_Label = []
        Target_Label_Flag = True
        input = pd.DataFrame(data=None)
        df = pd.DataFrame(data=None)

        mat = loadmat(mat_file, simplify_cells=True)  # load mat-file
        '''
        From Table-3 of Dataset paper (https://www.sciencedirect.com/science/article/pii/S235234092030562X#tbl0001), it's seen that, 
        Approaching (I1) interaction has 'Steady State' at the end & other interactions has 'Steady State' at first.
        Since we've to clip the WiFi packets to number of min_packets, we'll perform it from 'Steady State'.
        '''

        if len(mat['Raw_Cell_Matrix']) >= int(min_packets):
            if mat_file.find('_I1_T') > -1:
                # Crop from End
                ranges = [i for i in range(0, int(min_packets))]
            else:
                # Crop from Start
                ranges = [i for i in
                          range(len(mat['Raw_Cell_Matrix']) - int(min_packets), len(mat['Raw_Cell_Matrix']))]

        else:
            # Do padding: Copy number of lacking data from existing data to make data of equal min_packet length
            padding_length = int(min_packets) - int(len(mat['Raw_Cell_Matrix']))

            if mat_file.find('_I1_T') > -1:
                # Pad at the End
                ranges = [i for i in range(0, len(mat['Raw_Cell_Matrix']))]
                ranges.extend(ranges[-padding_length:])
            else:
                # Pad at Start
                ranges = [i for i in range(0, padding_length)]
                ranges.extend([i for i in range(0, len(mat['Raw_Cell_Matrix']))])

        for i in ranges:
            a = mat['Raw_Cell_Matrix'][i]  # Load Array

            if 'label' in a.keys():
                # print("Key exists")
                Target_Label.append(a['label'])
            else:
                # print("Key does not exist")
                Target_Label_Flag = False

            df_temp = pd.DataFrame(
                [a['timestamp_low'], a['noise'], a['agc'], a['RSSI_a'], a['RSSI_b'], a['RSSI_c']]).T
            df = pd.concat([df,
                            pd.concat([df_temp, pd.DataFrame(abs(a['CSI']).ravel()).T,
                                       pd.DataFrame(np.angle(a['CSI']).ravel()).T],
                                      axis=1, ignore_index=True)],
                           axis=0, ignore_index=True)
            # CSI Array (Abs value) of single Trial. Original Shape = (min_packets, NTx, NRx, Subcarriers)
            # CSI Array (Phase Angle value) of single Trial. Original Shape = (min_packets, NTx, NRx, Subcarriers)

        df.columns = cols
        Time_Diff = (df['timestamp_low'] - df['timestamp_low'].shift(1)).fillna(0)
        df.insert(1, 'Time_Diff', Time_Diff)
        df = df.drop(['timestamp_low'], axis=1)

        # Finalize each mat_file's content
        input = pd.concat([input, df], axis=0, ignore_index=True)

        # feature_scaler: Robust Scaler
        input = pd.DataFrame(self.sc.transform(input))
        input.columns = df.columns
        df = pd.DataFrame(data=None)

        if Target_Label_Flag:
            Target_Label = pd.DataFrame(Target_Label)
            Target_Label.columns = ['Target_Label']

        del mat_file, mat, i, a, df_temp, df, cols
        return input, Target_Label, Target_Label_Flag


    def classify(self, input, Target_Label, Target_Label_Flag):
        test_X = np.array(input).reshape(-1, self.min_packets, 366)

        flag = True
        for i in range(0, self.number_of_models):
            Y_pred_temp = self.model[i].predict(test_X, verbose = 0, batch_size = 12)
            Y_pred_temp = self.enc.inverse_transform(Y_pred_temp.reshape(-1, Y_pred_temp.shape[-1]))
            Y_pred_temp = np.array(pd.DataFrame(Y_pred_temp).iloc[:, 0].map(self.label_dict)).reshape(-1, 1)
            if flag:
                Y_pred = Y_pred_temp
                flag = False
            else:
                Y_pred = np.concatenate([Y_pred, Y_pred_temp], axis = 1)

        Y_pred, _ = scipy.stats.mode(Y_pred, axis = 1)
        Y_pred = self.smoothe_all(Y_pred, self.min_packets, self.mode_point, self.sliding_points)

        # Save Prediction in Folder
        if Target_Label_Flag:
            if not os.path.exists('.\\classified_labels{}\\Y_true'.format(date_time)):
                os.makedirs('.\\classified_labels{}\\Y_true'.format(date_time))

            Y_true = np.array(pd.DataFrame(Target_Label).iloc[:, 0].map(self.label_dict)).reshape(-1, 1)
            pd.DataFrame(Y_true).to_csv('.\\classified_labels{}\\Y_true\\{}.csv'.format(date_time, self.file_name), index=False)

        if not os.path.exists('.\\classified_labels{}\\Y_pred'.format(date_time)):
            os.makedirs('.\\classified_labels{}\\Y_pred'.format(date_time))
        pd.DataFrame(Y_pred).to_csv('.\\classified_labels{}\\Y_pred\\{}.csv'.format(date_time, self.file_name), index=False)

        saved_path = os.path.join(os.getcwd(), 'classified_labels{}'.format(date_time))
        return saved_path


    def COSTPMM(self, outputs, min_packets, mode_point, sliding_points):
        '''
        COSTPMM: Custom Output Smoothner, Ten-point Moving Mode
        mode_point = 20, sliding_points = 10 -------> Mode is taken for past & future 20 points of 10 intermediate points & rectified their (10 samples) value
        '''

        def repeater(value, repeated_points):
            value_repeat = []
            for i in range(0, repeated_points):
                value_repeat.extend(value)
            return value_repeat

        outputs_temp = repeater(list(scipy.stats.mode(outputs[:mode_point])[0].ravel()), mode_point)
        # print('outputs_temp = ', len(outputs_temp), ', expected =', sliding_points)

        for i in range(mode_point, len(outputs) - mode_point, sliding_points):
            # print('outputs_temp = ', len(outputs_temp), ', starting i =', i)
            past_mode = list(scipy.stats.mode(outputs[(i - mode_point):i])[0].ravel())
            future_mode = list(
                scipy.stats.mode(outputs[(i + sliding_points):(i + sliding_points + mode_point)])[0].ravel())

            if past_mode == future_mode:
                outputs_temp.extend(repeater(past_mode, sliding_points))  # or,  repeater(future_mode)
            else:
                # Zigzag prediction smoothner in the section of interaction change (Steady state to any other)
                past_half_points = int(np.round(sliding_points / 2))
                future_half_points = int(sliding_points - past_half_points)
                outputs_temp.extend(repeater(past_mode, past_half_points))  # past_half
                outputs_temp.extend(repeater(future_mode, future_half_points))  # future_half

                # Old. Delete if not necessary:     outputs_temp.extend(list(outputs[i : i+sliding_points].ravel()))

        remaining_points = abs(len(outputs_temp) - min_packets)
        outputs_temp.extend(repeater(list(scipy.stats.mode(outputs[- remaining_points:])[0].ravel()), remaining_points))

        # print('outputs_temp = ', len(outputs_temp), ', expected =', min_packets)
        return outputs_temp


    def smoothe_all(self, Y_pred, min_packets, mode_point, sliding_points):
        Y_pred_temp_list = []
        for mini_batch in range(0, len(Y_pred), min_packets):
            outputs = Y_pred[mini_batch: (mini_batch + min_packets)]
            Y_pred_temp_list.extend(self.COSTPMM(outputs, min_packets, mode_point, sliding_points))

        Y_pred_new = np.array(Y_pred_temp_list).reshape(-1, 1)

        return Y_pred_new


class background_Timer_Thread(QThread):
    def __init__(self):
        super(background_Timer_Thread, self).__init__()
        self.string = ['For the first-time run of this app, you have to be connected to internet for downloading',
                       'the requirements of Plotly HTML plot. From next-time, you do not need to be online.',
                       'If anytime you observe blank plot (white-screen), please enable internet connection and run the app.',
                       '... ... ...',
                       'Roll Mouse Wheel on the Dial (right-side of screen) to view different Trial Classifications.',
                       'If you observe multiple unit jump of Dial-Notch after single Roll of the Mouse Wheel,',
                       'Change your PC Mouse Wheel Settings like following: ',
                       'PC Settings > Mouse > Choose how many lines to scroll each time > Set the slider to 1',
                       '... ... ...',
                       'For the first-time run of this app, you have to be connected to internet for downloading',
                       'the requirements of Plotly HTML plot. From next-time, you do not need to be online.',
                       'If anytime you observe blank plot (white-screen), please enable internet connection and run the app.',
                       '... ... ...',
                       'Roll Mouse Wheel on the Dial (right-side of screen) to view different Trial Classifications.',
                       'If you observe multiple unit jump of Dial-Notch after single Roll of the Mouse Wheel,',
                       'Change your PC Mouse Wheel Settings like following: ',
                       'PC Settings > Mouse > Choose how many lines to scroll each time > Set the slider to 1',
                       'Copyright © 2023 | Developed by: Md. Mohi Uddin Khan | Article Authors: Md. Mohi Uddin Khan, Abdullah Bin Shams, Mohsin Sarker Raihan']

    # Define Signal
    update_Title_2_Text = pyqtSignal(str)

    def run(self):
        self.update_Title_2_Text.emit(self.string[0])

        for i in range(1, len(self.string)):
            time.sleep(3)
            self.update_Title_2_Text.emit(self.string[i])


class process_finished_delay_Thread(QThread):
    signal = pyqtSignal(int)
    def run(self):
        time.sleep(3)
        self.signal.emit(1)


# main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(HomePage())
    widget.resize(800, 500)

    # Make Frameless Window
    '''
    Qt.FramelessWindowHint ---> Makes Frameless Window
    Qt.WindowStaysOnTopHint ---> Makes Window-with Frame
    '''
    flags = Qt.WindowFlags(Qt.FramelessWindowHint)
    widget.setWindowFlags(flags)

    # Add Window Icon
    widget.setWindowIcon(QtGui.QIcon('Icon.ico'))

    '''
    Other Options:
    widget.showFullScreen()
    widget.showMaximized()
    
    widget.resize(1280, 800)       # (width, height)
    
    widget.setFixedWidth(1280)
    widget.setFixedHeight(800)
    '''
    # set the title
    widget.setWindowTitle("Human-to-Human Activity Recognizer from WiFi Signal")
    widget.show()

    sys.exit(app.exec_())
