from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import numpy as np

def plot_in_time(t, w, title="", xlabel="", ylabel=""):
    # t = np.linspace(0, 10, 5001)
    # w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    # plt.figure()
    plt.plot(t, w)
    title_plot= title
    xlabel_plot = xlabel
    ylabel_plot = ylabel
    plt.title(title_plot)
    plt.xlabel(xlabel)
    plt.grid(which='major')
    plt.show()

# This function combines chirp signals depening on the input parameters
# Duration parameter is used to specify the support of the time domain signal.
def chirpcombine(time, Aarray, fstartarray, fstoparray, typeflagarray,duration=10):
    num_of_chirps = len(Aarray)

    # Generate an empty array, later to be filled with chirps
    w = 0*chirp(time, f0=0, f1=0, t1=duration, method='linear')

    for i in range(num_of_chirps):
        w_current = Aarray[i]*chirp(time, f0=fstartarray[i], f1=fstoparray[i], t1=duration, method=typeflagarray[i])
        w = w_current + w
    return w
#


# # Uncommment this section to see the result of chirpcombine
# # A sprctrogram is used to display the results
#
# fs = 8000
# T = 5
# t = np.linspace(0, T, T*fs, endpoint=False)
# Amplitudes = [10, 10]
# fStartArray = [1500, 500]
# fStopArray = [1100, 1500]
# typeFlagArray = ['logarithmic', 'logarithmic']
#
# w = chirpcombine(time=t, Aarray=Amplitudes, fstartarray=fStartArray, fstoparray=fStopArray, typeflagarray=typeFlagArray, duration=T)
#
# ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512, nfft=2048)
#
# plt.figure()
# plot_in_time(t, w, "time domain graph of chirp","time", "amplitude")
#
#
# plt.figure()
# plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='twilight')
# plt.title('Spectrogram')
# plt.xlabel('t (sec)')
# plt.ylabel('Frequency (Hz)')
# plt.grid()
# plt.show()


