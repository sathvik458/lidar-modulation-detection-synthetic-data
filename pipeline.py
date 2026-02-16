import os
import random
import numpy as np
from scipy import signal
from scipy.signal import stft
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from models import build_resnet_1d, build_effnet


#Config 

CFG_GEN = {
    "NUM_SAMPLES": 4000,
    "FS": 100000,
    "DURATION": 0.02,
    "POSSIBLE_FREQS": [500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500],
    "MAX_SOURCES": 2,
    "MIN_AMPL": 0.3,
    "MAX_AMPL": 1.0,
    "NOISE_STD_RANGE": (0.005, 0.015),
    "MULTIPATH_PROB": 0.05,
    "WAVEFORMS": ["pwm", "sine", "triangle"],
    "RANDOM_SEED": 99,
}

CFG_TRAIN = {
    "BATCH_SIZE": 32,
    "EPOCHS": 35,
    "LR_1D": 1e-3,
    "LR_2D": 1e-4,
    "IMG_SIZE": (128, 128),
}

BASE_DIR = "lidar_pro_dataset"
SPECT_DIR = os.path.join(BASE_DIR, "spectrograms")

os.makedirs(SPECT_DIR, exist_ok=True)

CFG_GEN["T"] = int(CFG_GEN["FS"] * CFG_GEN["DURATION"])

random.seed(CFG_GEN["RANDOM_SEED"])
np.random.seed(CFG_GEN["RANDOM_SEED"])
tf.random.set_seed(CFG_GEN["RANDOM_SEED"])


#Signal generation
def generate_modulation(t, freq, waveform="pwm", duty=0.5, phase=0.0):
    if waveform == "pwm":
        return (signal.square(2 * np.pi * freq * t + phase, duty=duty) + 1) / 2
    elif waveform == "sine":
        return 0.5 * (1 + np.sin(2 * np.pi * freq * t + phase))
    elif waveform == "triangle":
        return 0.5 * (
            1 + signal.sawtooth(2 * np.pi * freq * t + phase, width=0.5)
        )
    return np.zeros_like(t)


def synthesize_sample():
    t = np.linspace(0, CFG_GEN["DURATION"], CFG_GEN["T"], endpoint=False)

    num_sources = random.randint(1, CFG_GEN["MAX_SOURCES"])
    freqs = random.sample(CFG_GEN["POSSIBLE_FREQS"], num_sources)

    combined = np.zeros_like(t)

    for f in freqs:
        amp = random.uniform(CFG_GEN["MIN_AMPL"], CFG_GEN["MAX_AMPL"])
        duty = random.uniform(0.2, 0.8)
        phase = random.uniform(0, 2 * np.pi)
        waveform = random.choice(CFG_GEN["WAVEFORMS"])

        combined += amp * generate_modulation(t, f, waveform, duty, phase)

    dist = random.uniform(5.0, 30.0)
    attenuation = 1.0 / (dist ** 2)
    clean = combined * attenuation * 10.0

    if random.random() < CFG_GEN["MULTIPATH_PROB"]:
        delay = int(random.uniform(10, 50))
        if delay < len(clean):
            clean[delay:] += clean[:-delay] * 0.3

    noise = np.random.normal(
        0, random.uniform(*CFG_GEN["NOISE_STD_RANGE"]), len(t)
    )

    signal_out = clean + noise
    signal_out = np.clip(signal_out, -1, 1)

    return signal_out.astype(np.float32), freqs

#Dataset Creation
def generate_dataset():
    num_samples = CFG_GEN["NUM_SAMPLES"]
    num_classes = len(CFG_GEN["POSSIBLE_FREQS"])

    X = np.zeros((num_samples, CFG_GEN["T"]), dtype=np.float32)
    Y = np.zeros((num_samples, num_classes), dtype=np.int8)

    print("Generating synthetic LiDAR dataset...")

    for i in tqdm(range(num_samples)):
        sig, freqs = synthesize_sample()
        X[i] = sig

        for f in freqs:
            idx = CFG_GEN["POSSIBLE_FREQS"].index(f)
            Y[i, idx] = 1

    return X, Y

#Spectrogram Creation
def generate_spectrograms(X):
    print("Generating spectrograms...")

    for i in tqdm(range(len(X))):
        f, t, Z = stft(
            X[i],
            fs=CFG_GEN["FS"],
            nperseg=256,
            noverlap=128,
        )

        S = np.abs(Z)
        S = np.log1p(S)

        S_img = (
            (S - S.min()) /
            (S.max() - S.min() + 1e-9) * 255
        ).astype(np.uint8)

        img = Image.fromarray(S_img)
        img = img.resize(CFG_TRAIN["IMG_SIZE"], Image.BILINEAR)

        img.save(os.path.join(SPECT_DIR, f"{i}.png"))

#Main execution
def main():
    X, Y = generate_dataset()
    generate_spectrograms(X)

    print("Dataset generation complete.")
    print(f"Samples: {len(X)}")
    print(f"Signal length: {X.shape[1]}")


if __name__ == "__main__":
    main()
