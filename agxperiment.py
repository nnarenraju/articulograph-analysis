# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
agxperiment.py
    Reads and plots information in .pos/.phn files
    input files are obtained from AG500 Articulograph

Created on Thu Oct 15 18:28:28 2020

__author__      = nnarenraju
__copyright__   = Copyright 2020, ag500-analysis
__credits__     = nnarenraju
__license__     = Apache License 2.0
__version__     = 1.0.0
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = inProgress


Documentation: NULL

"""

# extern
import os
import re
import glob
import logging
import argparse
import itertools
import numpy as np
import configparser
from scipy.io import wavfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 16})


class AG500():
    
    def __init__(self):
        # config.ini params
        self.config = None
        # Dataset directory absolute path
        self.data_dir = None
        
        # .pos file data (numsamples x 12 x 7)
        self.numsamples = 0
        
        # Channels
        self.channels  = None
        # Relationship plots
        self.relations = None
        
        # Data buffer
        self.posdata = None
        self.phndata = None
        self.wavdata = None
        
        # sampling rate (in Hz)
        self.pos_samplerate = 200
        self.wav_samplerate = 16000
        
        # row and col names ('e' is not req. for analysis)
        self.colnames = {'x':0, 'y':1, 'z':2, 'rho':3, 'theta':4, 'phi':5, 'e':6}
        
    def _parse_tuple(self, data):
        # Parsing an integer only tuple
        return tuple(k.strip() for k in data[1:-1].split(','))
    
    def _get_filenames(self, parent_path, extension):
        # get all filenames with a given extension and returns list
        os.chdir(parent_path)
        filenames = glob.glob("*.{}".format(extension))
        os.chdir(os.path.realpath(__file__))
        return filenames
    
    def read_pos(self, filepath=None):
        # Return 3D numpy array of .pos file data
        data = np.fromfile(filepath, dtype=np.float32)
        # set: numsamples x 12 x 7 format in all .pos files
        self.numsamples = int(len(data)/(12*7))
        # Reshape and return
        self.posdata = data.reshape((self.numsamples, 12, 7))
    
    def read_phn(self, filepath=None):
        # Return 2D numpy array of labels from .phn file
        with open(filepath, 'r') as foo:
            data = foo.read()    
        # Data clean-up
        data = re.findall("[0-9]+\s{1}[0-9]+\s{1}[a-zA-Z]+", data)
        # Reshape and return
        self.phndata = [line.split(" ") for line in data]
        
    def read_wav(self, filepath=None):
        # Return 2D numpy array of .wav file data
        self.wav_samplerate, self.wavdata = wavfile.read(filepath)
    
    def Stitch_speakers(self):
        # Supports only single channels stitch b/w speakers
        pass
    
    def Stitch_sounds(self):
        # Supports only single channels stitch b/w sounds
        pass
    
    def _figure(self):
        # nrows is the number of requested channels
        # ncols is the analysis axes (xy, yz, zx, ...)
        nrows = sum([val=='True' for val in self.channels.values()])
        ncols = len(self.relations)
        
        # Required row and col keys
        rows = np.where(list(self.channels.values()))[0]
        # col keys require special attention
        xidx = [self.colnames[val] for val in self.relations[:,1]]
        yidx = [self.colnames[val] for val in self.relations[:,0]]
        xlab = self.relations[:,1]
        ylab = self.relations[:,0]
        # Subplots
        if config.getboolean('OVERLAY', 'overlay'):
            szrows = 1
        else:
            szrows = nrows
        
        figure, axes = plt.subplots(szrows, ncols)
        figure.set_size_inches((8.0*ncols, 8.0*szrows))
        return (figure, axes, nrows, ncols, rows, xidx, yidx, xlab, ylab)
    
    def Stitch_channels(self, config, params, subject=None, session=None, filename=None, flag=-1):
        # Supports only single sounds unit stitch b/w channels
        # Get the start and end value from .phn data
        if config.getboolean('STITCH_CHANNELS', 'stitch_channels'):
            sound_unit = config['STITCH_CHANNELS']['sound_unit']
        elif config.getboolean('STITCH_SPEAKERS', 'stitch_speakers'):
            sound_unit = config['STITCH_SPEAKERS']['sound_unit']
            
        phn_index = [index for index, sound in enumerate(self.phndata) if sound_unit in sound]
        if not phn_index:
            logging.warning("sound_unit '{}' not found in given phn data".format(sound_unit))
            return
        else:
            phn_index = phn_index[0]
        
        start, end, label = self.phndata[phn_index]
        start = int(int(start)/(self.wav_samplerate / self.pos_samplerate))
        end = int(int(end)/(self.wav_samplerate / self.pos_samplerate))
        
        # Get figure params
        figure, axes, nrows, ncols, rows, xidx, yidx, xlab, ylab = params
        # Marker settings
        m = ['^', 'o'][flag]
        for nrow, ncol in itertools.product(range(nrows), range(ncols)):
            # rows[nrow] gives the channel number [0-11]
            # cols[ncol] gives the plot dims [xy, yz, zx, ...] as [0-5]
            x = self.posdata[start:end, rows[nrow], xidx[ncol]]
            y = self.posdata[start:end, rows[nrow], yidx[ncol]]
            
            # Setting axes
            if config.getboolean('OVERLAY', 'overlay'):
                cname = list(self.channels.keys())[rows[nrow]]
                axes[ncol].plot(x, y, '--', label="{}".format(cname), marker=m, fillstyle='none')
                axes[ncol].grid(True)
                axes[ncol].set_xlabel(xlab[ncol])
                axes[ncol].set_ylabel(ylab[ncol])
                axes[ncol].set_title("Sound '{}'".format(label))
                axes[ncol].legend()
            else:
                axes[nrow, ncol].plot(x, y, '--', marker=m, fillstyle='none')
                axes[nrow, ncol].grid(True)
                axes[nrow, ncol].set_xlabel(xlab[ncol])
                axes[nrow, ncol].set_ylabel(ylab[ncol])
                cname = list(self.channels.keys())[rows[nrow]]
                axes[nrow, ncol].set_title("Channel: {0}, Sound '{1}'".format(cname, label))
        
        # STITCH SPEAKERS BYPASS ROUTE
        if not abs(flag):
            return
        
        # Savefig
        # Make PDF option
        if self.make_PDF:
            ext = ".pdf"
        else:
            ext = ".png"
        
        if flag != 1:
            if not os.path.isdir('plots/stitch_channels'):
                os.makedirs('plots/stitch_channels')
            parent = "plots/stitch_channels/"
            figure.savefig(parent+"cstitch_{}-{}-{}{}".format(subject, session, filename, ext))
        else:
            if not os.path.isdir('plots/stitch_speakers'):
                os.makedirs('plots/stitch_speakers')
            parent = "plots/stitch_speakers/"
            figure.savefig(parent+"sstitch{}".format(ext))
        
        # Safety closure
        plt.close()
    
    def _get_parent(self, config, section):
        # Returns the parent dir PATH for given <section> in config
        Class    = config[section]['class']
        subject  = config[section]['subject']
        session  = config[section]['session']
        return (self.data_dir+"/"+Class+"/"+subject+"/"+session+"/", subject, session)
    
    def _get_files(self, config, parent):
        # Handing Filenames
        filenames = config['STITCH_CHANNELS']['filename']
        if not filenames:
            phn_files = self._get_filenames(parent+"phn_headMic/", "PHN")
            pos_files = self._get_filenames(parent+"pos/", "pos")
            # Sanity checks
            if len(phn_files) != len(pos_files):
                raise RuntimeError("Equal length expected for phn_files && pos_files")
            phn_names = [file[:4] for file in phn_files]
            pos_names = [file[:4] for file in pos_files]
            if len(list(set(phn_names) & set(pos_names))) != len(phn_files):
                raise NameError("Filenames expected to be the same b/w phn and pos dirs")
        else:
            filenames = [filenames]
            
        # Return all filenames as a list of names, NOT paths
        return filenames
    
    def analyse(self, config):
        # mkdirs
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        # Channels and relations
        self.channels = dict(config.items('CHANNEL'))
        # relationships required (stored as list of tuples)
        rdict = dict(config.items('PLOTTING'))
        self.relations = np.array([self._parse_tuple(tup) for tup in rdict.values()])
        self.data_dir = config['INPUT']['data_dir']
        
        # Stitch speakers
        if self.stitch_speakers:
            # Almost the same method as stitch channels
            params = self._figure()
            # Provide information about two speakers and a len threshold
            parent, subject_1, session_1 = self._get_parent(config, 'SPEAKER_1')
            session_1  = session_1[0] + session_1[-1]
            filenames_1 = self._get_files(config, parent)
            parent, subject_2, session_2 = self._get_parent(config, 'SPEAKER_2')
            session_2 = session_2[0] + session_2[-1]
            filenames_2 = self._get_files(config, parent)
            
            # Iterate through filenames and stitch channels
            filenames = filenames_1 + filenames_2
            subjects  = [subject_1] + [subject_2]
            for n, filename in enumerate(filenames):
                self.read_phn(parent+"phn_headMic/{}.PHN".format(filename))
                self.read_pos(parent+"pos/{}.pos".format(filename))
                # Get the required pos and phn files
                self.Stitch_channels(config, params, subject=subjects[n], flag=n)
        
        elif self.stitch_sounds:
            raise IOError("Under construction!")
            
        elif self.stitch_channels:
            # Read required files (phn, pos) from given information
            parent, subject, session = self._get_parent(config, 'STITCH_CHANNELS')
            # Get all filenames required
            filenames = self._get_files(config, parent)
            # prettifying savefig
            session = session[0] + session[-1]
            # Iterate through filenames and stitch channels
            params = self._figure()
            for filename in filenames:
                self.read_phn(parent+"phn_headMic/{}.PHN".format(filename))
                self.read_pos(parent+"pos/{}.pos".format(filename))
                # Get the required pos and phn files
                self.Stitch_channels(config, params, subject, session, filename)
            
    def set_config(self, config):
        # check config parser and get correct data-types
        # Boolean data-types
        self.stitch_channels = config.getboolean('STITCH_CHANNELS', 'stitch_channels')
        self.stitch_speakers = config.getboolean('STITCH_SPEAKERS', 'stitch_speakers')
        self.stitch_sounds = config.getboolean('STITCH_SOUNDS', 'stitch_sounds')
        self.make_PDF = config.getboolean('OUTPUT', 'make_PDF')
        
if __name__ == "__main__":
    
    # Argparse and Config handling
    p = argparse.ArgumentParser()
    # Default location of *.ini in working directory
    p.add_argument("-config", type=str, default='config.ini',
                   help='Configuration file for AG500 articulograph data analysis.')
    
    args = p.parse_args()
    # handle config.ini
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Logging sanity check
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Logging
    log_file = config['OUTPUT']['LOG_FILE']
    logging.basicConfig(filename=log_file, filemode='w', \
                        format='%(name)s@%(asctime)s - %(levelname)s - %(message)s',\
                        datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    
    logging.info("Start AG500 data analysis module")
    
    # Initialise module
    ag = AG500()
    # Set config params in object
    ag.set_config(config)
    
    # ANALYSIS
    ag.analyse(config)








