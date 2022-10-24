import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def freq2Note(f):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    return 12*(np.log2(f) - np.log2(a)) + 69

def note2Freq(note):
    # A4, MIDI index 69
    a = 440.0  # frequency of A (common value is 440Hz)
    #return 440*2**((n−69)/12)
    return (a / 32) * (2 ** ((note - 9) / 12.0))

def loiacono(y, subdivisionOfSemitone = 4.0, midistart = 0, midiend = 110, sr=44100):
    midilen   = midiend-midistart
    midiIndices = np.arange(midistart, midiend, 1/subdivisionOfSemitone)
    frequenciesHz = np.array([note2Freq(n) for n in midiIndices])
    wRadiansPerSample = 2*np.pi*frequenciesHz / sr
    
    result = np.zeros(int(len(wRadiansPerSample)), dtype=complex)
    for i, w in enumerate(wRadiansPerSample):
        complexPart = np.exp(-1j*w*np.arange(len(y)))
        result[i] = np.sum(np.multiply(y, complexPart ))
    return midiIndices, result

# import the file to be assessed
y, sr = librosa.load("KSHarp_A4_mf.wav", sr=None)
subdivisionOfSemitone = 4
midiIndices, result = loiacono(y, subdivisionOfSemitone=subdivisionOfSemitone, sr=sr)

#determine the note(s) present
notePattern = np.zeros(25*subdivisionOfSemitone)
zerothFreq = note2Freq(0)
for harmonic in range(1,5):
    hfreq = zerothFreq*harmonic
    hnote = freq2Note(hfreq)*subdivisionOfSemitone
    print("---")
    print(hfreq)
    print(hnote)
    if hnote < len(notePattern):
        notePattern[int(hnote)] = (1-(hnote%1))
        notePattern[int(hnote)+1] = (hnote%1)
        
#compressed = np.absolute(result)
thresh = 0.2
#factor = 0.03
#for i in range(len(compressed)):
#    if compressed[i] > thresh:
#        compressed[i] = thresh + (compressed[i] - thresh)*factor
        
peaks, properties = find_peaks(np.absolute(result), threshold=0.5, distance=2*subdivisionOfSemitone)
compressed = np.zeros(np.shape(np.absolute(result)))
for i in peaks:
    compressed[int(i)] = 1

#compressed[compressed<thresh] = 0
#compressed[compressed>=thresh] = 1
notes = np.correlate(notePattern, compressed, mode="same")
    
    
# using tuple unpacking for multiple Axes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.plot(notePattern)
ax2.plot(midiIndices, np.absolute(result))
ax3.plot(midiIndices, notes)
ax4.plot(midiIndices, compressed)
#plt.plot(midiIndices, np.absolute(result))
plt.show()

