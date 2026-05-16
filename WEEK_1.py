import csv
import numpy as np
from scipy.io import wavfile

channelResponse = []

# Open the CSV file
with open("WEEK_1_FILES/channel.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        # Check row exists and first column is not empty
        if row and row[0].strip() != "":
            channelResponse.append(float(row[0])) # Convert all data to floats to avoid weird errors


# Calculate DFT of channel response (for later use)
channelDFT = np.fft.fft(channelResponse, 1024)

# Open first WAV file
fs, samples = wavfile.read("WEEK_1_FILES/file01.wav")

print(fs)
print(samples)