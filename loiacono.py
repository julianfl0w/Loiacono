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
    # return 440*2**((note-69)/12)
    return (a / 32) * (2 ** ((note - 9) / 12.0))


def loadFlute():
    # import the file to be assessed
    # y, sr = librosa.load("../vsynth/samples/VSCO-2-CE/Woodwinds/Bassoon/sus/PSBassoon_A1_v1_1.wav", sr=None)
    # y, sr = librosa.load("white.wav", sr=None)
    # y, sr = librosa.load("../vsynth/samples/VSCO-2-CE/Brass/Trumpet/sus/Sum_SHTrumpet_sus_C3_v1_rr1.wav", sr=None)
    y, sr = librosa.load("ahh.wav", sr=None)
    return y, sr


class Loiacono:
    def __init__(
        self,
        subdivisionOfSemitone=4.0,
        midistart=30,
        midiend=110,
        sr=48000,
        multiple=50,
    ):

        # the dftlen is the period in samples of the lowest note, times the multiple
        # log ceiling
        lowestNoteNormalizedFreq = (note2Freq(midistart) / sr)
        #print(lowestNoteNormalizedFreq)
        #print(sr)
        self.m = multiple
        baseL2 = np.log2(multiple / lowestNoteNormalizedFreq)
        baseL2 = np.ceil(baseL2)
        #print(baseL2)
        self.DTFTLEN = int(2 ** baseL2)
        #print(self.DTFTLEN)
        self.SAMPLE_FREQUENCY = sr
        midilen = midiend - midistart
        self.midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
        frequenciesHz = np.array([note2Freq(n) for n in self.midiIndices])
        self.fprime = frequenciesHz / sr
        self.wRadiansPerSample = 2 * np.pi * self.fprime

        self.subdivisionOfSemitone = subdivisionOfSemitone
        self.midistart = midistart
        self.midiend = midiend
        self.getTwittleFactors(multiple=multiple)
        self.getHarmonicPattern()

    def getTwittleFactors(self, multiple):
        self.N = np.arange(self.DTFTLEN)
        self.normalizedFrequency = [
            note2Freq(note) / self.SAMPLE_FREQUENCY for note in self.midiIndices
        ]
        self.W = np.array([2 * np.pi * w for w in self.normalizedFrequency])

        self.WN = np.dot(np.expand_dims(self.W, 1), np.expand_dims(self.N, 0))
        self.EIWN = np.exp(-1j * self.WN)

        # each dtftlen should be an integer multiple of its period
        for i, fprime in enumerate(self.normalizedFrequency):
            dftlen = multiple / fprime
            # set zeros before the desired period (a multiple of pprime)
            self.EIWN[i, : int(self.DTFTLEN - dftlen)] = np.array([0])
            self.EIWN[i,:] /= dftlen**(1/2)
            
            #self.EIWN[i, int(self.DTFTLEN - dftlen) :] *= get_window(
            #    "hann", len(self.EIWN[i, int(self.DTFTLEN - dftlen) :])
            #)

    # function to generate note detection pattern
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

    def debugRun(self, y):
        nstart = time.time()
        self.run(y)
        nlen = time.time() - nstart
        print("nlen " + str(nlen))
                
    def run(self, y):

        startTime = time.time()
        result = np.dot(self.EIWN, y)
        endTime = time.time()
        # print("transfrom runtime (s) : " + str(endTime-startTime))
        self.absresult = np.absolute(result)
        
        self.findNote(self.absresult)

        # self.auto = np.correlate(y,y, mode="valid")

    def findNote(self, absresult):
        startTime = time.time()
        self.notes = np.correlate(absresult, self.notePattern, mode="valid")
        self.notesPadded = np.append(
            self.notes, np.zeros(int(len(self.notePattern) - 1))
        )
        endTime = time.time()
        # print("correlate runtime (s) : " + str(endTime-startTime))

        self.maxIndex = np.argmax(self.notesPadded)
        self.selectedNote = self.midistart + self.maxIndex / self.subdivisionOfSemitone
        self.selectedAmplitude = self.notesPadded[self.maxIndex]
        # print("selectedNote " + str(selectedNote))
        # print("expected " + str([selectedNote + h for h in self.hnotes]))

    def plot(self):
        # using tuple unpacking for multiple Axes
        fig, ((ax1)) = plt.subplots(1, 1)

        ax1.plot(self.midiIndices, self.absresult)
        #ax1.axis(ymin=0, ymax=max(self.absresult) + 1)
        # ax4.plot(self.auto)
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
                self.lpf = inertia * self.lpf + (1 - inertia) * self.absresult

        # using tuple unpacking for multiple Axes
        fig, ((ax1)) = plt.subplots(1, 1)
        ax1.plot(self.lpf)
        ax1.axis(ymin=0, ymax=max(self.lpf) + 1)
        plt.show()

    def squareTest(self):
        y, sr = librosa.load("square220.wav", sr=None)
        y = y[int(len(y) / 2) : int(len(y) / 2 + DTFTLEN)]
        linst.run(y)
        print(linst.selectedNote)
        linst.plot()


if __name__ == "__main__":

    m = 10
    if sys.argv[1] == "whiteNoiseTest":
        linst = Loiacono(
            sr=48000, midistart=30, midiend=128, subdivisionOfSemitone=2.0, multiple=m
        )
        linst.whiteNoiseTest()

    else:
        infile = sys.argv[1]
    # load the wav file
    y, sr = librosa.load(infile, sr=None)
    # generate a Loiacono based on this SR
    linst = Loiacono(
        sr=sr, midistart=30, midiend=128, subdivisionOfSemitone=2.0, multiple=m
    )
    
    # get a section in the middle of sample for processing
    y = y[int(len(y) / 2) : int(len(y) / 2 + linst.DTFTLEN)]
    linst.run(y)
    print(linst.selectedNote)
    linst.plot()
