import numpy as np

class Autotune:
    def __init__(self):
        self.ref_freqs = [
            65.41,
            82.41,
            110.00,
            146.83,
            196.00,
            246.94,
            329.63,
            440.00,
            587.33,
            783.99,
            1046.50,
        ]
        self.note_dict = self.generate_interpolated_frequencies()

    def generate_interpolated_frequencies(self):
        note_dict = []
        for i in range(len(self.ref_freqs) - 1):
            freq_low = self.ref_freqs[i]
            freq_high = self.ref_freqs[i + 1]
            interpolated_freqs = np.linspace(freq_low, freq_high, num=10, endpoint=False)
            note_dict.extend(interpolated_freqs)
        note_dict.append(self.ref_freqs[-1])
        return note_dict

    def autotune_f0(self, f0):
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = closest_note
        return autotuned_f0
