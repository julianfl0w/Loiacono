import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import get_window


def freq2Note(f):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    return 12 * (np.log2(f) - np.log2(a)) + 69


def note2Freq(note):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    # return 440*2**((n−69)/12)
    return (a / 32) * (2 ** ((note - 9) / 12.0))


def loiacono(y, subdivisionOfSemitone=4.0, midistart=0, midiend=110, sr=44100):
    midilen = midiend - midistart
    midiIndices = np.arange(midistart, midiend, 1 / subdivisionOfSemitone)
    frequenciesHz = np.array([note2Freq(n) for n in midiIndices])
    wRadiansPerSample = 2 * np.pi * frequenciesHz / sr

    result = np.zeros(int(len(wRadiansPerSample)), dtype=complex)
    for i, w in enumerate(wRadiansPerSample):
        complexPart = np.exp(-1j * w * np.arange(len(y)))
        result[i] = np.sum(np.multiply(y, complexPart))
    return midiIndices, result


# import the file to be assessed
y, sr = librosa.load("LDFlute_susvib_C4_v1_2.wav", sr=None)
subdivisionOfSemitone = 1
# y = y*get_window(window="hamming", Nx=len(y))
midistart = 20
midiend = 200

midiIndices, result = loiacono(
    y, midistart=midistart, midiend=midiend,subdivisionOfSemitone=subdivisionOfSemitone, sr=sr
)

# determine the note(s) present
notePattern = np.zeros(50 * subdivisionOfSemitone)
zerothFreq = note2Freq(0)
hnotes = []
for harmonic in range(1, 5):
    hfreq = zerothFreq * harmonic
    hnote = freq2Note(hfreq) * subdivisionOfSemitone
    print("---")
    print(hfreq)
    print(hnote)
    if hnote + 1 < len(notePattern):
        hnotes += [hnote]
        notePattern[int(hnote)] = 1 - (hnote % 1)
        notePattern[int(hnote) + 1] = hnote % 1
# notePattern = np.convolve(notePattern, get_window(window="hamming", Nx=5), mode="same")
# compressed = np.absolute(result)
thresh = 0.2
# factor = 0.03
# for i in range(len(compressed)):
#    if compressed[i] > thresh:
#        compressed[i] = thresh + (compressed[i] - thresh)*factor
peaks, properties = find_peaks(
    np.absolute(result), threshold=0.5, distance=2 * subdivisionOfSemitone
)
compressed = np.zeros(np.shape(np.absolute(result)))
for i in peaks:
    compressed[int(i)] = 1

# compressedNew = compressed.copy()
# compressedNew += np.roll(compressed,-1)
# compressedNew += np.roll(compressed,1)
# compressed = compressedNew

absresult = np.absolute(result)
print(peaks)

peak_sum_max = 0
selectedNote = 0
for p in peaks:
    peak_sum = 0
    for o in hnotes:
        if round(p + o) < len(absresult):
            peak_sum += absresult[round(p + o)]
    if peak_sum > peak_sum_max:
        peak_sum_max = peak_sum
        selectedNote = p

# compressed[compressed<thresh] = 0
# compressed[compressed>=thresh] = 1
notes = np.correlate(absresult, notePattern, mode="valid")
# notes = np.append(np.zeros(int(len(notePattern)/2)), notes)
notes = np.append(notes, np.zeros(int(len(notePattern)-1)))

selectedNote = midistart+np.argmax(notes)
print("selectedNote " + str(selectedNote))
print("expected " + str([selectedNote + h for h in hnotes]))

# using tuple unpacking for multiple Axes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.plot(notePattern)
ax2.plot(midiIndices, np.absolute(result))
ax3.plot(midiIndices, notes)
ax4.plot(compressed)
# plt.plot(midiIndices, np.absolute(result))
plt.show()
