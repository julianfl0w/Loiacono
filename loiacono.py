import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import get_window
import time

def freq2Note(f):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    return 12 * (np.log2(f) - np.log2(a)) + 69


def note2Freq(note):
    # A4, MIDI index 69
    #a = 440.0  # frequency of A (common value is 440Hz)
    return 440*2**((note-69)/12)
    #return (a / 32) * (2 ** ((note - 9) / 12.0))

def loadFlute():
    # import the file to be assessed
    y, sr = librosa.load("LDFlute_susvib_C4_v1_2.wav", sr=None)
    return y, sr

class Loiacono:

            
    def __init__(self, subdivisionOfSemitone=4.0, midistart=0, midiend=110, sr=44100, DTFTLEN=512):
        self.DTFTLEN = DTFTLEN
        self.SAMPLE_FREQUENCY = sr
        midilen = midiend - midistart
        self.WCOUNT = midilen*subdivisionOfSemitone
        self.midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
        frequenciesHz = np.array([note2Freq(n) for n in self.midiIndices])
        
        self.wRadiansPerSample = 2 * np.pi * frequenciesHz / sr

        self.subdivisionOfSemitone = subdivisionOfSemitone
        self.midistart = midistart
        self.midiend = midiend
        self.getTwittleFactors()
        self.getHarmonicPattern()
        
    def getTwittleFactors(self):
        self.N = np.arange(self.DTFTLEN)
        self.W = np.array([2 * np.pi * note2Freq(note)/self.SAMPLE_FREQUENCY for note in np.arange(self.WCOUNT)/self.subdivisionOfSemitone])
        
        self.WN = np.dot(np.expand_dims(self.W,1),np.expand_dims(self.N, 0))
        self.EIWN = np.exp(-1j*self.WN)
        
    def getHarmonicPattern(self):
        
        # create the note pattern
        # only need to do this once
        self.notePattern = np.zeros(int(50 * self.subdivisionOfSemitone))
        zerothFreq = note2Freq(0)
        self.hnotes = []
        for harmonic in range(1, 5):
            hfreq = zerothFreq * harmonic
            hnote = freq2Note(hfreq) * self.subdivisionOfSemitone
            if hnote + 1 < len(self.notePattern):
                self.hnotes += [hnote]
                self.notePattern[int(hnote)] = 1 - (hnote % 1)
                self.notePattern[int(hnote) + 1] = hnote % 1

    
    def run(self, y):

        startTime = time.time()
        result = np.dot(self.EIWN, y)        
        endTime = time.time()
        print("transfrom runtime (s) : " + str(endTime-startTime))
        self.absresult = np.absolute(result)
        self.findNote(self.absresult)

    def findNote(self, absresult):
        startTime = time.time()
        self.notes = np.correlate(absresult, self.notePattern, mode="valid")
        self.notes = np.append(self.notes, np.zeros(int(len(self.notePattern)-1)))
        endTime = time.time()
        print("correlate runtime (s) : " + str(endTime-startTime))

        selectedNote = self.midistart+np.argmax(self.notes)/self.subdivisionOfSemitone
        print("selectedNote " + str(selectedNote))
        print("expected " + str([selectedNote + h for h in self.hnotes]))

    def plot(self):
        # using tuple unpacking for multiple Axes
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
        ax1.plot(self.notePattern)
        ax2.plot(self.midiIndices, self.absresult)
        ax3.plot(self.notes)
        # plt.plot(midiIndices, np.absolute(result))
        plt.show()
        
if __name__ == "__main__":
    y, sr = loadFlute()
    DTFTLEN = 256    
    y = y[int(len(y)/2): int(len(y)/2+DTFTLEN)]
    linst = Loiacono(sr=sr, DTFTLEN=DTFTLEN, midiend=256, subdivisionOfSemitone=1.0)
    linst.run(y)
    #linst.plot()

