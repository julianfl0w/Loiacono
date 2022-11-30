import numpy as np
import librosa
import matplotlib.pyplot as plt
import time
import sys

class Loiacono:
    def __init__(
        self,
        fprime,
        multiple=50,
    ):
        # the dftlen is the period in samples of the lowest note, times the multiple
        # log ceiling
        lowestNoteNormalizedFreq = fprime[0]
        #print(lowestNoteNormalizedFreq)
        #print(sr)
        self.m = multiple
        baseL2 = np.log2(multiple / lowestNoteNormalizedFreq)
        baseL2 = np.ceil(baseL2)
        #print(baseL2)
        self.DTFTLEN = int(2 ** baseL2)
        #print(self.DTFTLEN)
        self.fprime = fprime

        self.wRadiansPerSample = 2 * np.pi * self.fprime

        self.getTwittleFactors(multiple=multiple)
        getHarmonicPattern()

    def getTwittleFactors(self, multiple):
        self.N = np.arange(self.DTFTLEN)
        self.W = np.array([2 * np.pi * w for w in self.fprime])

        self.WN = np.dot(np.expand_dims(self.W, 1), np.expand_dims(self.N, 0))
        self.EIWN = np.exp(-1j * self.WN)

        # each dtftlen should be an integer multiple of its period
        for i, fprime in enumerate(self.fprime):
            dftlen = multiple / fprime
            # set zeros before the desired period (a multiple of pprime)
            self.EIWN[i, : int(self.DTFTLEN - dftlen)] = np.array([0])
            self.EIWN[i,:] /= dftlen**(1/2)

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
        
        findNote(
            spectrum = self.absresult
        )

        # self.auto = np.correlate(y,y, mode="valid")

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
