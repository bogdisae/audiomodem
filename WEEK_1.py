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
print(f"Shape of samples: {np.shape(samples)}")
print(samples)



# THIS FUNCTION RETURNS A LIST OF BLOCKS.
# WHERE BLOCK[0] IS FIRST CYCLIC PREFIX, BLOCK [1] IS FIRST IDFT, BLOCK[2] IS SECOND PREFIX ETC
def split_pattern(x):
    i = 0
    blocks = []
    pattern = [32, 1024]

    p = 0
    n = len(x)

    while i < n:
        size = pattern[p % 2]
        blocks.append(x[i:i+size])
        i += size
        p += 1

    return blocks


def channel_equalise(block, h):
    Y = np.fft.fft(block)          # 1024-point DFT
    H = np.fft.fft(h, 1024)        # channel IR zero-padded to 1024
    X = Y / H                       # equalise all bins
    data_bins = X[1:512]
    return data_bins


#print(split_pattern(samples)[0])

blocks = split_pattern(samples)

for i in range(0, len(blocks), 2):
    prefix = blocks[i]
    data_block = blocks[i+1]

    equalised_data = channel_equalise(data_block, channelResponse)
    if i == 0:
        print(np.shape(equalised_data))
        print(equalised_data[0:10])