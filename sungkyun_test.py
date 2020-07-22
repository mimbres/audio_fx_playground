# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Wed Jul 22 18:04:41 2020
@author: skchang@cochlear.ai
"""
# pip install pydub, gin, crepe
import wavio # for writing wav file
import matplotlib.pyplot as plt # for plot
import numpy as np
import tensorflow as tf
from ddsp import core
from ddsp import processors
from ddsp import synths
from ddsp import effects

# load audio sample
fp = './pop.00003.wav'
x, fs = tf.audio.decode_wav(tf.io.read_file(fp)) # x: (T, 1)
fs = int(fs)
x = tf.transpose(x) # x: (1, T)
x = x[:, :32000]
wavio.write('x.wav', x.numpy(), 8000, sampwidth=2)
#%% Exponential decay reverb
reverb1 = effects.ExpDecayReverb(reverb_length=8000)
gain = [[20.]] # gain: Linear gain of impulse response
decay = [[0.1]] 
# decay: Exponential decay coefficient. The final impulse response is
#          exp(-(2 + exp(decay)) * time) where time goes from 0 to 1.0 over the
#          reverb_length samples.
x_out = reverb1(x, gain, decay)
wavio.write('x_edrev.wav', x_out[0,:].numpy(), 8000, sampwidth=2)

#%%  Noise reverb
reverb2 = effects.FilteredNoiseReverb(reverb_length=8000, scale_fn=None)

# gaussian filtered band pass.
n_frames = 1000
n_frequencies = 100
frequencies = np.linspace(0, fs / 2.0, n_frequencies)
center_frequency = 300.0 * np.linspace(0, 1.0, n_frames)
width = 50.0
gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)

# Actually make the magnitudes.
magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
magnitudes = magnitudes[np.newaxis, ...]
magnitudes /= magnitudes.sum(axis=-1, keepdims=True) * 5

x_out = reverb2(x, magnitudes)
wavio.write('x_nsrev.wav', x_out[0,:].numpy(), 8000, sampwidth=2)
plt.matshow(np.rot90(magnitudes[0]), aspect='auto')
plt.title('Frequency Response'); plt.xlabel('Time'); plt.ylabel('Frequency')

#%% FIR Filter: bandpath filter
fir_filter = effects.FIRFilter(scale_fn=None)

n_seconds = x.shape[1] / fs
frame_rate = 100  # Hz
n_frames = int(n_seconds * frame_rate)
n_samples = int(n_frames * fs / frame_rate)

n_frequencies = 100
frequencies = np.linspace(0, fs / 2.0, n_frequencies)

center_frequency = 3000 + np.zeros(n_frames)
width = 1000.0
gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)

magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
magnitudes = magnitudes[np.newaxis, ...]
magnitudes /= magnitudes.max(axis=-1, keepdims=True)

x_out = fir_filter(x, magnitudes)
wavio.write('x_fir.wav', x_out[0,:].numpy(), 8000, sampwidth=2)
plt.matshow(np.rot90(magnitudes[0]), aspect='auto')

#%% FIR Filter: sweep
fir_filter = effects.FIRFilter(scale_fn=None)

n_seconds = x.shape[1] / fs
frame_rate = 100  # Hz
n_frames = int(n_seconds * frame_rate)
n_samples = int(n_frames * fs / frame_rate)

n_frequencies = 100
frequencies = np.linspace(0, fs / 2.0, n_frequencies)

lfo_rate = 5 # Hz
n_cycles = n_seconds * lfo_rate
center_frequency = 800 + 500 * np.sin(np.linspace(0, 2.0*np.pi*n_cycles, n_frames))
width = 800.0
gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)

magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
magnitudes = magnitudes[np.newaxis, ...]
magnitudes /= magnitudes.max(axis=-1, keepdims=True)

x_out = fir_filter(x, magnitudes)
wavio.write('x_fir2_sweep.wav', x_out[0,:].numpy(), 8000, sampwidth=2)
plt.matshow(np.rot90(magnitudes[0]), aspect='auto')
plt.title('Frequency Response'); plt.xlabel('Time'); plt.ylabel('Frequency')


#%% Mod Delay
def sin_phase(mod_rate):
    """Helper function."""
    n_samples = x.shape[-1]
    n_seconds = n_samples / fs
    phase = tf.sin(tf.linspace(0.0, mod_rate * n_seconds * 2.0 * np.pi, n_samples))
    return phase[tf.newaxis, :, tf.newaxis]

def modulate_audio(audio, center_ms, depth_ms, mod_rate):
    mod_delay = effects.ModDelay(center_ms=center_ms,
                                  depth_ms=depth_ms,
                                  gain_scale_fn=None,
                                  phase_scale_fn=None)
    
    phase = sin_phase(mod_rate)  # Hz
    gain = 1.0 * np.ones_like(audio)[..., np.newaxis]
    return 0.5 * mod_delay(audio, gain, phase)

x_out = modulate_audio(x, center_ms=0.75, depth_ms=0.75, mod_rate=0.25)
wavio.write('x_phase_mod1.wav', x_out[0,:].numpy(), 8000, sampwidth=2)

x_out = modulate_audio(x, center_ms=25.0, depth_ms=1.0, mod_rate=2.0)
wavio.write('x_phase_mod2.wav', x_out[0,:].numpy(), 8000, sampwidth=2)

x_out = modulate_audio(x, center_ms=25.0, depth_ms=12.5, mod_rate=5.0)
wavio.write('x_phase_mod3.wav', x_out[0,:].numpy(), 8000, sampwidth=2)
