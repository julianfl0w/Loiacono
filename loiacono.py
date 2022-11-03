import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import get_window
import time
import sys

def freq2Note(f):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    return 12 * (np.log2(f) - np.log2(a)) + 69


def note2Freq(note):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    #return 440*2**((note-69)/12)
    return (a / 32) * (2 ** ((note - 9) / 12.0))

def loadFlute():
    # import the file to be assessed
    #y, sr = librosa.load("../vsynth/samples/VSCO-2-CE/Woodwinds/Bassoon/sus/PSBassoon_A1_v1_1.wav", sr=None)
    #y, sr = librosa.load("white.wav", sr=None)
    #y, sr = librosa.load("../vsynth/samples/VSCO-2-CE/Brass/Trumpet/sus/Sum_SHTrumpet_sus_C3_v1_rr1.wav", sr=None)
    y, sr = librosa.load("ahh.wav", sr=None)
    return y, sr

class Loiacono:

            
    def __init__(self, subdivisionOfSemitone=4.0, midistart=0, midiend=110, DTFTLEN=512, sr=48000):
        self.DTFTLEN = DTFTLEN
        self.SAMPLE_FREQUENCY = sr
        midilen = midiend - midistart
        self.midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
        frequenciesHz = np.array([note2Freq(n) for n in self.midiIndices])
        
        self.wRadiansPerSample = 2 * np.pi * frequenciesHz / sr

        self.subdivisionOfSemitone = subdivisionOfSemitone
        self.midistart = midistart
        self.midiend = midiend
        self.getTwittleFactors()
        self.getHarmonicPattern()
        
    def getTwittleFactors(self, multiple = 24):
        self.N = np.arange(self.DTFTLEN)
        self.WUnity = [note2Freq(note)/self.SAMPLE_FREQUENCY for note in self.midiIndices]
        self.W = np.array([2 * np.pi * w for w in self.WUnity])
        
        self.WN = np.dot(np.expand_dims(self.W,1),np.expand_dims(self.N, 0))
        self.EIWN = np.exp(-1j*self.WN)
        
        # each dtftlen should be an integer multiple of its period
        for i, wu in enumerate(self.WUnity):
            dftlen = multiple / wu
            self.EIWN[i,:int(self.DTFTLEN-dftlen)] = np.array([0])
            #self.EIWN[i,:] /= dftlen
            self.EIWN[i,int(self.DTFTLEN-dftlen):] *= get_window("hann", len(self.EIWN[i,int(self.DTFTLEN-dftlen):]))
        
        
        
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
        #print("transfrom runtime (s) : " + str(endTime-startTime))
        self.absresult = np.absolute(result)
        self.findNote(self.absresult)
        
        #self.auto = np.correlate(y,y, mode="valid")

    def findNote(self, absresult):
        startTime = time.time()
        self.notes = np.correlate(absresult, self.notePattern, mode="valid")
        self.notesPadded = np.append(self.notes, np.zeros(int(len(self.notePattern)-1)))
        endTime = time.time()
        #print("correlate runtime (s) : " + str(endTime-startTime))

        self.maxIndex = np.argmax(self.notesPadded)
        self.selectedNote = self.midistart+self.maxIndex/self.subdivisionOfSemitone
        self.selectedAmplitude = self.notesPadded[self.maxIndex]
        #print("selectedNote " + str(selectedNote))
        #print("expected " + str([selectedNote + h for h in self.hnotes]))

    def plot(self):
        # using tuple unpacking for multiple Axes
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
        ax1.plot(self.notePattern)
        ax2.plot(self.midiIndices, self.absresult)
        ax3.plot(self.notes)
        #ax4.plot(self.auto)
        # plt.plot(midiIndices, np.absolute(result))
        plt.show()
    
    def whiteNoiseTest(self):
        self.lpf = None
        inertia = 0.99
        for i in range(10000):
            y = np.random.randn((self.DTFTLEN))
            self.sr = 48000
            self.run(y)
            if self.lpf is None:
                self.lpf = self.absresult
            else:
                self.lpf = inertia*self.lpf + (1-inertia)*self.absresult
                
        # using tuple unpacking for multiple Axes
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(self.lpf)
        plt.show()
    
    def squareTest(self):
        y, sr = librosa.load("square220.wav", sr=None)
        y = y[int(len(y)/2): int(len(y)/2+DTFTLEN)]
        linst.run(y)
        print(linst.selectedNote)
        linst.plot()
        
    def noteIDTest(self, infile):
        y, sr = librosa.load(infile, sr=None)
        if sr != self.SAMPLE_FREQUENCY:
            raise Exception("Sample frequency mismatch! got " + str(sr) + " expected " + str(self.SAMPLE_FREQUENCY))
        y = y[int(len(y)/2): int(len(y)/2+DTFTLEN)]
        linst.run(y)
        print(linst.selectedNote)
        linst.plot()

        
        
if __name__ == "__main__":
    DTFTLEN = 2**14
    linst = Loiacono(sr=44100, DTFTLEN=DTFTLEN, midistart=0, midiend=128, subdivisionOfSemitone=2.0)
    linst.noteIDTest(sys.argv[1])