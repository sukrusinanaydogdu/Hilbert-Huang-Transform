import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, chirp
from my_utils import*

def ApplyBoundaryCondition(signal):
    # This function is used to get rid of errors in the boudaries
    # It basically equates first value to the first defined (not nan) number
    # and it equates the last value to the last defined number
    # By doing so, errors in the edgers are avoided.

    for i in range(len(signal)):
        if not np.isnan(signal[i]):
            signal[0] = signal[i]
            break
    signal_reverse = signal[::-1]
    for i in range(len(signal_reverse)):
        if not np.isnan(signal_reverse[i]):
            signal[-1] = signal_reverse[i]
            break
    return signal

def is_Residual(amplitude):
    # This function is used to check whether a time series data can be considered as a residual or not
    # To be residual, the function should not have any maxima or minima
    # Residual functions can be constant or monotonic
    # This code can be further improved because it is adapdet from another function in here, which makes it
    # computationally expensive.
    nodata_flag = np.nan
    envelope_max = np.zeros(len(amplitude))
    envelope_min = np.zeros(len(amplitude))
    envelope_max[:] = nodata_flag
    envelope_min[:] = nodata_flag

    minima_total = 0
    maxima_total = 0
    graph_stat_history = 'up'
    graph_stat_recent = 'up'
    for i in range(len(amplitude)):
        if i == 1:
            if amplitude[i] > amplitude[i - 1]:
                graph_stat_recent = 'up'
                graph_stat_history = 'up'
            elif amplitude[i] < amplitude[i - 1]:
                graph_stat_recent = 'down'
                graph_stat_history = 'down'
            else:
                graph_stat_recent = graph_stat_history
            continue
        if amplitude[i] > amplitude[i - 1]:
            graph_stat_recent = 'up'
        elif amplitude[i] < amplitude[i - 1]:
            graph_stat_recent = 'down'
        else:
            graph_stat_recent = graph_stat_history

        if graph_stat_recent != graph_stat_history:
            if graph_stat_recent == 'up':
                envelope_min[i] = amplitude[i - 1]
                minima_total += 1
                graph_stat_history = 'up'
            elif graph_stat_recent == 'down':
                envelope_max[i] = amplitude[i - 1]
                maxima_total += 1
                graph_stat_history = 'down'

    return (maxima_total == 0) or (minima_total == 0)

def is_IMF(amplitude):
    # This funciton is used to check whether a time seriesdata is an IMF or not.
    # To be IMF, difference of # of minimas and # of maximas should be 1 or 0
    # Also the function should be zero mean but this check is done in the overall loop
    nodata_flag = np.nan
    envelope_max = np.zeros(len(amplitude))
    envelope_min = np.zeros(len(amplitude))
    envelope_max[:] = nodata_flag
    envelope_min[:] = nodata_flag

    minima_total = 0
    maxima_total = 0
    graph_stat_history = 'up'
    graph_stat_recent = 'up'
    for i in range(len(amplitude)):
        if i == 1:
            if amplitude[i] > amplitude[i - 1]:
                graph_stat_recent = 'up'
                graph_stat_history = 'up'
            elif amplitude[i] < amplitude[i - 1]:
                graph_stat_recent = 'down'
                graph_stat_history = 'down'
            else:
                graph_stat_recent = graph_stat_history
            continue
        if amplitude[i] > amplitude[i - 1]:
            graph_stat_recent = 'up'
        elif amplitude[i] < amplitude[i - 1]:
            graph_stat_recent = 'down'
        else:
            graph_stat_recent = graph_stat_history

        if graph_stat_recent != graph_stat_history:
            if graph_stat_recent == 'up':
                envelope_min[i] = amplitude[i - 1]
                minima_total += 1
                graph_stat_history = 'up'
            elif graph_stat_recent == 'down':
                envelope_max[i] = amplitude[i - 1]
                maxima_total += 1
                graph_stat_history = 'down'

    return (np.abs(maxima_total - minima_total) < 2)

def IMF_check(amplitude):
    # This function checks the numbers of maxima and minima and prints them
    # It was a basis for the maxima/minima analysis so all the code in other functions is the same with that one
    nodata_flag = np.nan
    envelope_max = np.zeros(len(amplitude))
    envelope_min = np.zeros(len(amplitude))
    envelope_max[:] = nodata_flag
    envelope_min[:] = nodata_flag

    minima_total = 0
    maxima_total = 0
    graph_stat_history = 'up'
    graph_stat_recent = 'up'
    for i in range(len(amplitude)):
        if i == 1:
            if amplitude[i] > amplitude[i - 1]:
                graph_stat_recent = 'up'
                graph_stat_history = 'up'
            elif amplitude[i] < amplitude[i - 1]:
                graph_stat_recent = 'down'
                graph_stat_history = 'down'
            else:
                graph_stat_recent = graph_stat_history
            continue
        if amplitude[i] > amplitude[i - 1]:
            graph_stat_recent = 'up'
        elif amplitude[i] < amplitude[i - 1]:
            graph_stat_recent = 'down'
        else:
            graph_stat_recent = graph_stat_history

        if graph_stat_recent != graph_stat_history:
            if graph_stat_recent == 'up':
                envelope_min[i] = amplitude[i - 1]
                minima_total += 1
                graph_stat_history = 'up'
            elif graph_stat_recent == 'down':
                envelope_max[i] = amplitude[i - 1]
                maxima_total += 1
                graph_stat_history = 'down'

    print('Total minima of my function', minima_total)
    print('Total maxima of my function', maxima_total)

def getEnvelopes(amplitude):
    # Given atime series data, this function return the upper and lower envelope points (not functions) of it
    nodata_flag = np.nan
    envelope_max = np.zeros(len(amplitude))
    envelope_min = np.zeros(len(amplitude))
    envelope_max[:] = nodata_flag
    envelope_min[:] = nodata_flag

    minima_total = 0
    maxima_total = 0
    graph_stat_history = 'up'
    graph_stat_recent = 'up'
    for i in range(len(amplitude)):
        if i == 1:
            if amplitude[i] > amplitude[i - 1]:
                graph_stat_recent = 'up'
                graph_stat_history = 'up'
            elif amplitude[i] < amplitude[i - 1]:
                graph_stat_recent = 'down'
                graph_stat_history = 'down'
            else:
                graph_stat_recent = graph_stat_history
            continue
        if amplitude[i] > amplitude[i - 1]:
            graph_stat_recent = 'up'
        elif amplitude[i] < amplitude[i - 1]:
            graph_stat_recent = 'down'
        else:
            graph_stat_recent = graph_stat_history

        if graph_stat_recent != graph_stat_history:
            if graph_stat_recent == 'up':
                envelope_min[i] = amplitude[i - 1]
                minima_total += 1
                graph_stat_history = 'up'
            elif graph_stat_recent == 'down':
                envelope_max[i] = amplitude[i - 1]
                maxima_total += 1
                graph_stat_history = 'down'

    return [envelope_min, envelope_max]


def CubicEnvelope(time, envelope_points, applyBC = False):
    # Given envelope points, the funciton interpolates with cubic spline and returns the envelope functions.
    all_minima_point = []
    all_minima_point_time = []
    all_maxima_point = []
    all_maxima_point_time = []

    for i in range(len(envelope_points)):
        if not np.isnan(envelope_points[i]):
            all_minima_point.append(envelope_points[i])
            all_minima_point_time.append(time[i])



    all_minima_point = np.asarray(all_minima_point)
    all_minima_point_time = np.asarray(all_minima_point_time)

    if (applyBC):
        for i in range(len(envelope_points)):
            if not np.isnan(envelope_points[i]):
                envelope_points[0] = envelope_points[i]
                break
        envelope_points_reverse = envelope_points[::-1]
        for i in range(len(envelope_points_reverse)):
            if not np.isnan(envelope_points_reverse[i]):
                envelope_points[-1] = envelope_points_reverse[i]
                break

    if((len(all_minima_point_time) == 0) or (len(all_minima_point_time) == 0)):
        return np.zeros(len(envelope_points))
    print('size of all_minima_point_time = {} all_minima_point = {}'.format(len(all_minima_point_time), len(all_minima_point)))
    cs_minima = CubicSpline(all_minima_point_time, all_minima_point)
    envelope_minima_cs = cs_minima(time_test)

    # plt.figure()
    # plt.title('Inside CS MİNİMA CASE')
    # plt.plot(time_test, envelope_minima_cs)
    return envelope_minima_cs


# Settings of experiments
fs= 100
frequency_1 = 1
frequency_2 = 4
total_time_test = 5
res_test = 1

time_test = np.arange(0, total_time_test*fs, res_test) / fs
amplitude_1 = np.sin(2 * np.pi * frequency_1 * total_time_test * (time_test/time_test[-1]))
amplitude_2 = np.sin(2 * np.pi * frequency_2 * total_time_test * (time_test/time_test[-1]))
amplitude_fusion = amplitude_1 + amplitude_2

Amplitudes = [10, 10]
fStartArray = [1500, 500]
fStopArray = [1100, 1500]
typeFlagArray = ['linear', 'linear']

input = chirpcombine(time=time_test, Aarray=Amplitudes, fstartarray=fStartArray, fstoparray=fStopArray, typeflagarray=typeFlagArray, duration=total_time_test)
# input = amplitude_fusion
iter_no = 0
input_functon = input
input_for_test = input
residual = []
IMF_array = []

plt.figure()
plt.title('Input ')
# plt.ylim([-1, 1])
plt.plot(time_test, input)
plt.show()

while(True):
    [envelope_min_test, envelope_max_test] = getEnvelopes(input)

    envelope_min_test = ApplyBoundaryCondition(envelope_min_test)
    envelope_max_test = ApplyBoundaryCondition(envelope_max_test)
    # print('size of envelope_min_test = {} envelope_max_test = {}'.format(len(envelope_min_test), len(envelope_max_test)))

    # plt.figure()
    # plt.title('Envelopes for debug before')
    # plt.plot(time_test, input)
    # plt.plot(time_test, envelope_min_test, 'bo')
    # plt.plot(time_test, envelope_max_test, 'ro')
    # plt.show()
    # plt.close()

    envelope_minima_cs = CubicEnvelope(time_test, envelope_min_test)
    envelope_maxima_cs = CubicEnvelope(time_test, envelope_max_test)

    # plt.figure()
    # plt.title('Signals and maximas')
    # plt.plot(time_test, input)
    # plt.plot(time_test, envelope_min_test, 'bo')
    # plt.plot(time_test, envelope_max_test, 'ro')
    # plt.show()
    # plt.close()

    mid_value = np.zeros(len(envelope_maxima_cs))
    for i in range(len(mid_value)):
        mid_value[i] = (envelope_maxima_cs[i] + envelope_minima_cs[i]) / 2

    extracted_val = input - mid_value

    mean = np.mean(extracted_val)

    print('mean of extracted value:', mean)
    # print(' If abs(min and max) < 2 then it is an IMF')
    print('Iteration no: {}'.format(iter_no))
    IMF_check(extracted_val)
    #
    # plt.figure()
    # plt.title('Figure Iter')
    # plt.plot(time_test, extracted_val)
    # plt.show()
    # plt.close()
    iter_no += 1
    if (is_Residual(extracted_val) or (np.abs(mean) < 1e-7)):
        residual.append(extracted_val)
        # plt.figure()
        # plt.title('Residual')
        # plt.plot(time_test, extracted_val)
        # plt.ylim([-1, 1])
        # plt.show()
        # plt.close()
        break
    elif(is_IMF(extracted_val) and (np.abs(np.mean(extracted_val)) < 1e-2)):
       IMF_array.append(extracted_val)
       input_functon = input_functon - extracted_val
       input = input_functon
       continue
    else:
        input = input - extracted_val
        continue

total_sum = residual[0]
# plotting loop
for i in range(len(IMF_array)):
    plt.figure()
    plt.title('IMF number {}'.format(i))
    plt.plot(time_test, IMF_array[i])
    plt.show()
    total_sum  = total_sum + IMF_array[i]

plt.figure()
plt.title('Residual final ')
plt.ylim([-1, 1])
plt.plot(time_test, residual[0])
plt.show()

# plt.figure()
# plt.title('Total Sum ')
# # plt.ylim([-1, 1])
# plt.plot(time_test, total_sum)
# plt.show()

error = np.sum(np.abs(total_sum - input_for_test))
print('error = {}'.format(error))
# num_IMF = len(IMF_array)
# result_matrix = np.zeros((num_IMF, len(time_test) -1))
#
# for i in range(num_IMF):
#
#     hil_transform = hilbert(IMF_array[i])
#     amplitude_envelope = np.abs(hil_transform)
#     instantaneous_phase = np.unwrap(np.angle(hil_transform))
#     instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
#
#     result_matrix[i:] = np.abs(hil_transform[1:])
#     plt.figure()
#     plt.plot(time_test ,amplitude_envelope[0:])

# plt.figure()
# plt.pcolormesh(time_test[1:], instantaneous_frequency, result_matrix[1:], cmap='twilight')
# plt.title('Spectrogram')
# plt.xlabel('t (sec)')
# plt.ylabel('Frequency (Hz)')
# plt.grid()
# plt.show()
# plt.figure()
# plt.plot()
# # plt.ylim([-5, 5])
# plt.imshow(result_matrix, vmin=0, vmax=255)

#
# t = time_test
# # signal = amplitude_fusion
# signal = chirp(t, 20.0, t[-1], 100.0)
# signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
#
# analytic_signal = hilbert(signal)
# amplitude_envelope = np.abs(analytic_signal)
# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * fs)
#
# fig = plt.figure()
# ax0 = fig.add_subplot(211)
# ax0.plot(t, signal, label='signal')
# ax0.plot(t, amplitude_envelope, label='envelope')
# ax0.set_xlabel("time in seconds")
# ax0.legend()
# ax1 = fig.add_subplot(212)
# ax1.plot(t[1:], instantaneous_frequency)
# ax1.set_xlabel("time in seconds")
# plt.show()


# ax1.set_ylim(0.0, 120.0)
