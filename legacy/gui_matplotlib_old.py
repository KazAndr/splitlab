import os
import sys

import your

import numpy as np

from astropy import units as u

import matplotlib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QCheckBox
from PyQt5.QtWidgets import QLabel, QPushButton, QFrame, QVBoxLayout
from PyQt5.QtWidgets import QGridLayout, QTextEdit, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget, QSpinBox

from PyQt5.QtGui import QIcon, QGuiApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt


matplotlib.use('Qt5Agg')
matplotlib.rc('xtick', labelsize=5)
matplotlib.rc('ytick', labelsize=5)
matplotlib.rcParams.update({'font.size': 5})


class PlotCanvas(FigCanvas):

    def __init__(self, title, index, data, dd_data):

        self.data = data
        self.dd_data = dd_data

        self.fig = Figure(figsize=(5, 7), dpi=300)

        FigCanvas.__init__(self, self.fig)
        self.setParent(None)

        FigCanvas.updateGeometry(self)

        widths = [8, 4]
        heights = [4, 8, 8]
        spec5 = self.fig.add_gridspec(
            ncols=2,
            nrows=3,
            width_ratios=widths,
            height_ratios=heights
        )

        ax = self.fig.add_subplot(spec5[0, 0])  # row, col
        ax.set_title(
            "{0}\nsubint {1}".format(title, index)
            )

        chart = np.sum(self.dd_data, axis=0)

        ax.plot(chart / np.max(chart), color='black', lw=0.5)
        ax.set_xlim(0, len(self.dd_data[0]))
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Normalized intensity')

        ax = self.fig.add_subplot(spec5[1, 0])
        ax.imshow(self.dd_data, aspect='auto')
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('Frequency channels')

        ax = self.fig.add_subplot(spec5[2, 0])
        ax.imshow(self.data.T, aspect='auto')
        ax.set_xlabel('Time samples')
        ax.set_ylabel('Frequency channels')

        ax = self.fig.add_subplot(spec5[1, 1])

        pow_by_chnl = np.sum(np.transpose(self.dd_data), axis=0)[::-1]

        ax.plot(
            pow_by_chnl / np.max(pow_by_chnl),
            range(len(pow_by_chnl)),
            color='black',
            lw=0.5)
        ax.set_ylim(0, len(pow_by_chnl))
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Normalized intensity')

        self.fig.subplots_adjust(wspace=0.05, hspace=0.05)

        self.show()


class WarningBox(QMessageBox):

    def __init__(self, title, text):
        super().__init__()
        self.title = title
        self.text = text
        self.initUI()

    def initUI(self):
        self.setIcon(QMessageBox.Warning)
        self.setWindowTitle('Warning!')
        self.setText('{0}\n\n{1}'.format(self.title, self.text))
        self.setStandardButtons(QMessageBox.Ok)
        self.setDefaultButton(QMessageBox.Ok)


class QuestionBox(QMessageBox):

    def __init__(self, title, text):
        super().__init__()
        self.title = title
        self.text = text
        self.initUI()

    def initUI(self):
        self.setIcon(QMessageBox.Question)
        self.setWindowTitle('Question!')
        self.setText(self.title)
        self.setInformativeText(self.text)
        self.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        self.setDefaultButton(QMessageBox.Yes)


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Labeling data'
        self.left = 20
        self.top = 30

        #  Deafault values for header
        self.n_param_lines = 11
        self.n_spectra = None
        self.n_channels = None
        self.f_high = None
        self.f_low = None
        self.t_sample = None
        self.steps = []

        sizeObject = QDesktopWidget().screenGeometry(-1)

        self.width = sizeObject.height()
        self.height = sizeObject.width()
        self.initUI()

    def initUI(self):
        '''
        Using the main parametrs of the window
        '''
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowIcon(QIcon('icons{0}logo.png'.format(os.sep)))

        ######################################################################

        self.label = QLabel('No label')
        self.label.setObjectName('label')
        self.label.setStyleSheet('QLabel#label {font: bold;}')
        self.label.setAlignment(Qt.AlignCenter)

        layout_label = QGridLayout()
        layout_label.addWidget(self.label, 0, 0, 1, 1)

        self.header_box = QLabel(self.get_mesage_for_header())
        self.header_box.setObjectName('header_box')
        self.header_box.setAlignment(Qt.AlignLeft)

        layout_header_box = QVBoxLayout()
        layout_header_box.addWidget(
            self.header_box, 0, Qt.AlignTop | Qt.AlignLeft)

        self.dm_label = QLabel('DM')
        self.size_label = QLabel('N t_samples')
        self.name = QLabel('Object name')

        self.name_value = QLineEdit()
        self.name_value.setText('Object name')

        self.dm_value = QLineEdit()
        self.dm_value.setValidator(QDoubleValidator(1, 9999, 4))
        self.dm_value.setText('56.758')

        self.window_value = QLineEdit()
        self.window_value.setValidator(QIntValidator(1, 1024))
        self.window_value.setText('256')

        layout_editable_values = QGridLayout()
        # addWidget(QWidget, row, column, rows, columns)
        layout_editable_values.addWidget(self.name, 0, 0, 1, 1)
        layout_editable_values.addWidget(self.name_value, 0, 1, 1, 1)
        layout_editable_values.addWidget(self.dm_label, 1, 0, 1, 1)
        layout_editable_values.addWidget(self.dm_value, 1, 1, 1, 1)
        layout_editable_values.addWidget(self.size_label, 2, 0, 1, 1)
        layout_editable_values.addWidget(self.window_value, 2, 1, 1, 1)

        self.labeling_box = QTextEdit()
        self.labeling_box.append('Subint: label')
        self.labeling_box.setReadOnly(True)
        self.labeling_box.verticalScrollBar()

        layout_labeling_box = QGridLayout()
        layout_labeling_box.addWidget(self.labeling_box, 0, 0, 1, 1)

        self.use_mask = QCheckBox("Using a freq. mask")
        self.use_mask.setChecked(True)
        self.auto_incr = QCheckBox("Auto. increment")
        self.auto_incr.setChecked(True)

        layout_check_box = QGridLayout()
        layout_check_box.addWidget(self.use_mask, 0, 0, 1, 1)
        layout_check_box.addWidget(self.auto_incr, 1, 0, 1, 1)

        self.next_button = QPushButton('Next scan')
        self.previous_button = QPushButton('Previous scan')
        self.spinBox = QSpinBox(self)
        self.spinBox.setRange(0, 10)
        self.goto_button = QPushButton('Go to')

        layout_navigation = QGridLayout()
        # addWidget(QWidget, row, column, rows, columns)
        layout_navigation.addWidget(self.spinBox, 0, 0, 1, 1)
        layout_navigation.addWidget(self.goto_button, 0, 1, 1, 1)
        layout_navigation.addWidget(self.next_button, 1, 1, 1, 1)
        layout_navigation.addWidget(self.previous_button, 1, 0, 1, 1)

        self.to_pulse = QPushButton('Pulse (P)')
        self.to_RFI = QPushButton('RFI (I)')
        self.to_None = QPushButton('None (O)')
        self.to_pulse_and_RFI = QPushButton('Pulse and RFI')

        layout_labelibg = QGridLayout()
        # addWidget(QWidget, row, column, rows, columns)
        layout_labelibg.addWidget(self.to_pulse, 0, 0, 1, 1)
        layout_labelibg.addWidget(self.to_RFI, 0, 1, 1, 1)
        layout_labelibg.addWidget(self.to_None, 1, 0, 1, 1)
        layout_labelibg.addWidget(self.to_pulse_and_RFI, 1, 1, 1, 1)

        self.start = QPushButton('Start labeling')
        self.stop = QPushButton('Stop labeling')

        layout_process = QGridLayout()
        # addWidget(QWidget, row, column, rows, columns)
        layout_process.addWidget(self.stop, 0, 0, 1, 1)
        layout_process.addWidget(self.start, 0, 1, 1, 1)

        self.save_file = QPushButton('Save log file')
        self.save_image = QPushButton('Save current image')

        layout_save_image = QGridLayout()
        # addWidget(QWidget, row, column, rows, columns)
        layout_save_image.addWidget(self.save_file, 0, 0, 1, 2)
        layout_save_image.addWidget(self.save_image, 1, 0, 1, 2)

        # deacivate several buttons before loading data
        self.next_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.to_pulse.setEnabled(False)
        self.to_RFI.setEnabled(False)
        self.to_None.setEnabled(False)
        self.to_pulse_and_RFI.setEnabled(False)
        self.stop.setEnabled(False)
        self.spinBox.setEnabled(False)
        self.goto_button.setEnabled(False)
        self.save_file.setEnabled(False)
        self.save_image.setEnabled(False)
        self.labeling_box.setEnabled(False)
        self.use_mask.setEnabled(False)
        self.auto_incr.setEnabled(False)

        # plot random data
        self.plotter = PlotCanvas(
            'TEST',
            0,
            np.random.rand(256, 256),
            np.random.rand(256, 256)
        )

        self.layout_plot = QGridLayout()
        self.layout_plot.addWidget(self.plotter, 0, 0, 1, 1)

        self.next_button.clicked.connect(self.replot_next)
        self.next_button.setShortcut("Ctrl+Right")
        self.next_button.setToolTip("Hotkeys: Ctrl+Right")
        self.previous_button.clicked.connect(self.replot_previous)
        self.previous_button.setShortcut("Ctrl+Left")
        self.previous_button.setToolTip("Hotkeys: Ctrl+Left")
        self.start.clicked.connect(self.start_labeling)
        self.stop.clicked.connect(self.stop_labeling)
        self.goto_button.clicked.connect(self.goto)
        self.goto_button.setShortcut("Ctrl+Return")
        self.goto_button.setToolTip("Hotkeys: Ctrl+Enter")
        self.to_pulse.clicked.connect(lambda: self.set_label('Pulse'))
        self.to_pulse.setShortcut("P")
        self.to_pulse.setToolTip("Hotkey: p")
        self.to_RFI.clicked.connect(lambda: self.set_label('RFI'))
        self.to_RFI.setShortcut("I")
        self.to_RFI.setToolTip("Hotkey: i")
        self.to_None.clicked.connect(lambda: self.set_label('None'))
        self.to_None.setShortcut("O")
        self.to_None.setToolTip("Hotkey: o")
        self.to_pulse_and_RFI.clicked.connect(
            lambda: self.set_label('Pulse and RFI')
        )

        self.save_file.clicked.connect(self.save_labeling_results)
        self.save_file.setShortcut("Ctrl+S")
        self.save_file.setToolTip("Hotkeys: Ctrl+S")
        self.save_image.clicked.connect(self.save_current_image)
        self.save_image.setShortcut("Ctrl+I")
        self.save_image.setToolTip("Hotkeys: Ctrl+I")
        self.use_mask.setShortcut("Ctrl+M")
        self.use_mask.setToolTip("Hotkeys: Ctrl+M")

        '''
        Компановка рабочих областей на виджеты
        '''

        frame_label = QFrame()
        frame_label.setLayout(layout_label)

        frame_header_box = QFrame()
        frame_header_box.setLayout(layout_header_box)

        frame_editable_values = QFrame()
        frame_editable_values.setLayout(layout_editable_values)

        frame_labeling_box = QFrame()
        frame_labeling_box.setLayout(layout_labeling_box)

        frame_check_box = QFrame()
        frame_check_box.setLayout(layout_check_box)

        frame_navigation = QFrame()
        frame_navigation.setLayout(layout_navigation)

        frame_labelibg = QFrame()
        frame_labelibg.setLayout(layout_labelibg)

        frame_process = QFrame()
        frame_process.setLayout(layout_process)

        self.frame_plot = QFrame()
        self.frame_plot.setObjectName('frame_plot')
        self.frame_plot.setLayout(self.layout_plot)

        frame_save_image = QFrame()
        frame_save_image.setLayout(layout_save_image)

        self.hbox = QGridLayout()
        # addWidget(QWidget, row, column, rows, columns)
        self.hbox.addWidget(frame_label, 0, 3, 1, 2)
        self.hbox.addWidget(frame_header_box, 1, 3, 1, 2)
        self.hbox.addWidget(frame_editable_values, 2, 3, 1, 2)
        self.hbox.addWidget(frame_labeling_box, 3, 3, 3, 2)
        self.hbox.addWidget(frame_check_box, 6, 3, 1, 2)
        self.hbox.addWidget(frame_navigation, 7, 3, 1, 2)
        self.hbox.addWidget(frame_labelibg, 8, 3, 1, 2)
        self.hbox.addWidget(frame_process, 9, 3, 1, 2)
        self.hbox.addWidget(frame_save_image, 11, 3, 1, 2)
        self.hbox.addWidget(self.frame_plot, 0, 0, 13, 1)

        self.main_panel = QWidget()
        self.main_panel.setLayout(self.hbox)
        self.setCentralWidget(self.main_panel)
        self.show()

    # Button functions

    def start_labeling(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        self.fil_file, _ = QFileDialog.getOpenFileName(
                    self, 'Choose file',
                    '', 'Filterbank file (*.fil)',
                    options=options)
        if self.fil_file:

            # loading data

            self.DM = float(self.dm_value.text())
            self.window = int(self.window_value.text())

            self.object = self.load_data(self.fil_file)
            self.get_header_info()

            self.steps = range(0, self.n_spectra, self.window)
            self.spinBox.setRange(0, len(self.steps))

            self.labels = self.get_labels()

            # self.refresh_labelbox()

            f_hi = self.f_high / 1000
            f_lo = self.f_low / 1000

            self.freq_list = np.linspace(f_lo, f_hi, self.n_channels)
            delays_list = self.delays_DM_list(self.freq_list, self.DM)
            tsamp = tsamp = self.t_sample * u.second

            self.delays_list_point = [
                int(round(i, 0))
                for i in (delays_list / tsamp.to(u.millisecond)).value
            ]

            self.mask = self.get_mask()

            # acivate  buttons after loading data

            self.next_button.setEnabled(True)
            self.previous_button.setEnabled(True)
            self.to_pulse.setEnabled(True)
            self.to_RFI.setEnabled(True)
            self.to_None.setEnabled(True)
            self.to_pulse_and_RFI.setEnabled(True)
            self.stop.setEnabled(True)
            self.spinBox.setEnabled(True)
            self.goto_button.setEnabled(True)
            self.save_file.setEnabled(True)
            self.save_image.setEnabled(True)
            self.labeling_box.setEnabled(True)
            self.use_mask.setEnabled(True)
            self.auto_incr.setEnabled(True)

            # deactivation boxes for input DM and w_size values

            self.dm_value.setEnabled(False)
            self.window_value.setEnabled(False)
            self.name_value.setEnabled(False)

            # nulling data_index
            self.data_index = 0

            self.header_box.setText(self.get_mesage_for_header())

            self.replot_next()
        else:
            msg = WarningBox(
                'File has not uploaded.',
                ('For activation all functions of this '
                 'program one has to upload a filterbank file!')
            )
            msg.exec_()

    def replot_next(self):
        step = self.steps[self.data_index]
        self.data = self.object.get_data(nstart=step, nsamp=self.window)

        if self.use_mask.isChecked():
            self.data = self.data * self.mask
        else:
            pass

        self.dd_data = self.dedispersion(
            self.data.T,
            self.delays_list_point[::-1]
        )

        self.plotter = PlotCanvas(
            os.path.basename(self.fil_file),
            self.data_index,
            self.data,
            self.dd_data
        )

        self.layout_plot.addWidget(self.plotter, 0, 0, 1, 1)
        self.frame_plot.setLayout(self.layout_plot)
        self.hbox.addWidget(self.frame_plot, 0, 0, 13, 1)
        self.main_panel.setLayout(self.hbox)
        self.setCentralWidget(self.main_panel)

        self.label.setText(str(self.labels[self.data_index][1]))

        self.show()

        self.data_index += 1

    def replot_previous(self):
        self.data_index -= 2
        self.replot_next()

    def goto(self):
        self.data_index = self.spinBox.value()
        self.replot_next()

    def set_label(self, label):
        if self.labels[self.data_index - 1][1] != 'No label':
            msg = QuestionBox(
                'This subint has already labeled.',
                'Would you like to relabel this subint?'
            )
            reply = msg.exec_()

            if reply == QMessageBox.No:
                return None

        self.labels[self.data_index - 1][1] = label
        self.label.setText(label)
        # self.refresh_labelbox()
        
        if self.auto_incr.isChecked():
            self.replot_next()

    def save_current_image(self):
        path_dir = os.path.join(
            os.getcwd(),
            '{0}_{1}'.format(self.name_value.text(), self.window)
        )

        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)

        filename = '{0}_subint_{1}.png'.format(
            os.path.basename(self.fil_file)[:-4],
            self.data_index - 1
        )

        self.plotter.fig.savefig('{0}{1}{2}'.format(
            path_dir,
            os.sep,
            filename
            ), dpi=350)

    # process functions

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return idx

    def get_mask(self):
        F_HI_MASK = 1.48
        F_LO_MASK = 1.26

        start = self.find_nearest(self.freq_list, F_LO_MASK)
        end = self.find_nearest(self.freq_list, F_HI_MASK)

        hi_mask_lines = start
        low_mask_lines = self.n_channels - end
        payload_lines = self.n_channels - low_mask_lines - hi_mask_lines

        hi_mask = np.full((hi_mask_lines, self.window), 0)
        payload = np.full((payload_lines, self.window), 1)
        low_mask = np.full((low_mask_lines, self.window), 0)

        mask = np.concatenate((hi_mask, payload, low_mask))

        return mask.T

    def get_mesage_for_header(self):
        msg = (
            'Header info:\n' +
            '\n' +
            'n_spectra: {0}\n'.format(self.n_spectra) +
            'n_channels: {0}\n'.format(self.n_channels) +
            'f_high: {0}\n'.format(self.f_high) +
            'f_low: {0}\n'.format(self.f_low) +
            't_sample: {0}\n'.format(self.t_sample) +
            'N subints: {0}\n'.format(len(self.steps))
        )

        return msg
    
    """
    def refresh_labelbox(self):
        self.labeling_box.clear()
        temp_array = []
        for line in self.labels:
            if line[1] == 'No label':
                temp_array.append('subint {0}: {1}'.format(line[0], line[1]))
            else:
                temp_array.append(
                    '*** subint {0}: {1} ***'.format(line[0], line[1])
                )

        self.labeling_box.append('\n'.join(temp_array))
    """

    def get_header_info(self):

        self.n_spectra = self.object.your_header.nspectra
        self.n_channels = self.object.your_header.nchans
        self.t_sample = self.object.your_header.tsamp

        fch1 = self.object.your_header.fch1
        bw = self.object.your_header.bw

        temp_f = fch1 + bw

        if fch1 > temp_f:
            self.f_high = fch1
            self.f_low = temp_f
        else:
            self.f_high = temp_f
            self.f_low = fch1

    def get_labels(self):
        labeling_log = '{0}{1}labeling_log'.format(os.getcwd(), os.sep)
        if not os.path.isdir(labeling_log):
            os.mkdir('labeling_log')

        name_part = os.path.basename(self.fil_file)[:-4]
        end_of_name = '_{0}_{1}.logcsv'.format(
            self.name_value.text(),
            self.window
        )

        pathname = 'labeling_log{0}{1}{2}'.format(
            os.sep,
            name_part,
            end_of_name
        )

        if os.path.isfile(pathname):
            labels = []
            with open(pathname, 'r') as file:
                # skip header
                [file.readline() for _ in range(self.n_param_lines)]

                for line in file:
                    labels.append(line.strip().split(','))
        else:
            labels = [[i, 'No label'] for i in range(len(self.steps))]

        return labels

    def t_func_DM_mu(self, DM, mu):
        return 4.148808 * DM * mu**(-2)  # DM[cm-3pc], mu[GHz]

    def delays_DM(self, f_lo, f_hi, DM):
        """
        \deltat = 4.148808 ms * [(f_lo)^-2[GHz] -  (f_hi)^-2[GHz]] * DM[cm-3pc]
        """
        return round(4.148808 * ((f_lo)**-2 - (f_hi)**-2) * DM, 2)

    def delays_DM_list(self, f_list, DM):

        dt_list = []
        for v in f_list:

            t1 = self.t_func_DM_mu(DM, float(v))
            t2 = self.t_func_DM_mu(DM, float(f_list[-1]))

            dt_list.append(round(t1 - t2, 1))

        return dt_list

    def load_data(self, filename):
        my_object = your.Your(filename)

        return my_object

    def dedispersion(self, array, delays_list):
        new_array = []
        for line, points in zip(array, delays_list):
            new_array.append(np.roll(line, -points))
        return new_array

    def stop_labeling(self):
        self.next_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.to_pulse.setEnabled(False)
        self.to_RFI.setEnabled(False)
        self.to_None.setEnabled(False)
        self.to_pulse_and_RFI.setEnabled(False)
        self.stop.setEnabled(False)
        self.spinBox.setEnabled(False)
        self.goto_button.setEnabled(False)
        self.goto_button.setEnabled(False)
        self.labeling_box.setEnabled(False)
        self.save_file.setEnabled(False)
        self.save_image.setEnabled(False)
        self.use_mask.setEnabled(False)
        self.auto_incr.setEnabled(False)

        self.dm_value.setEnabled(True)
        self.window_value.setEnabled(True)
        self.name_value.setEnabled(True)

        self.save_labeling_results()

    def save_labeling_results(self):
        name_part = os.path.basename(self.fil_file)[:-4]
        end_of_name = '_{0}_{1}.logcsv'.format(
            self.name_value.text(),
            self.window
        )

        pathname = 'labeling_log{0}{1}{2}'.format(
            os.sep,
            name_part,
            end_of_name
        )

        with open(pathname, 'w') as file:
            file.write('# Header information\n')
            file.write('# filename: {}\n'.format(self.fil_file))
            file.write('# N_spectra: {}\n'.format(self.n_spectra))
            file.write('# f_high [MHz]: {}\n'.format(self.f_high))
            file.write('# f_low [MHz]: {}\n'.format(self.f_low))
            file.write('# N_channels: {}\n'.format(self.n_channels))
            file.write('# DM [cm-3pc]: {}\n'.format(self.DM))
            file.write('# t_sample [s]: {}\n'.format(self.t_sample))
            file.write('# t_samples_in_window: {}\n'.format(self.window))
            file.write('# N_subints: {}\n'.format(len(self.steps)))
            file.write('\n')

            for line in self.labels:
                file.write('{0},{1}\n'.format(line[0], line[1]))

        msg = WarningBox('Log file was saved.', pathname)
        msg.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
