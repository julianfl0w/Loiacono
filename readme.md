**The Loiacono Transform**

The Loiacono Transform modifies the Discrete Time Fourier Transform such that the frequency bins span 12-Tone Equal Temperment (TET) evenly. For example, each note (A, A#, B...) may be analyzed as 100 cents. This is opposed to the Discrete Fourier Transform (DFT), which is evenly spaced across frequency, thereby giving substantially lower resolution for notes of lower pitch. The Loiacono Transform is necessarily slower than the Fast Fourier Transform (FFT) (the standard implementation of the DFT), but this disadvantage is offset by modern hardware.

The Loiacono Transform finds primary application in music analysis. It was developed for use in a vocoder. 

**Mathematical Definition**
Function f(x) = 440*2**((x−69)/12)

let p = 1000
s = 
\textbf{ω} = f(m), m = {0,1,2...p}

Let y be an audio signal of length n. Then the Loiacono Transform is defined as:
L(y, ω) = X(ω)
For ω in \textbf{ω} 

Where the DTFT is defined as
$$X(ω) = \sum_{n=-∞}^∞ x[n]e^{-iωn} $$
