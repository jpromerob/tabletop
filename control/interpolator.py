import numpy as np
import pdb

def interpolate_repeated_values(signal):
    if signal[-1]==signal[-2]:
        signal[-1] += 0.00001
    i = 0
    fail_count = 0
    while i < len(signal):

        if signal[i] == signal[i-1]:
            j = i
            while j < len(signal)-1 and signal[j] == signal[i]:
                j+=1
            steps = j-i
            # print(f"Interpolating between {signal[i-1]} and {signal[j]}  with {steps} steps")
            # print(signal[i:j])
            delta = (signal[j]-signal[i-1])/steps
            try:
                signal[i:j] = np.arange(signal[i-1], signal[j], delta)
            except:
                fail_count += steps
                pass
            i = j-1
        i += 1
        print(fail_count)
    return signal

# Example array with repeated values
signal = np.ones((20))

# Interpolate repeated values
interpolated_signal = interpolate_repeated_values(signal)
print(interpolated_signal)
