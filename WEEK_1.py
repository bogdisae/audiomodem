import csv
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def plot_argand_complex(data):

    x = np.asarray(data.real)
    y = np.asarray(data.imag)

    quadrants = np.where(
        (x >= 0) & (y >= 0),
        0,
        np.where(
            (x < 0) & (y >= 0),
            1,
            np.where((x < 0) & (y < 0), 2, 3)
        )
    )

    quadrant_colors = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red"])
    point_colors = quadrant_colors[quadrants]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=point_colors, s=18, alpha=0.85, edgecolors="none")

    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Argand Diagram of Equalised Data")
    plt.grid()
    plt.axis('equal')
    plt.show()

def channel_equalise(block, h):
    Y = np.fft.fft(block)          # 1024-point DFT
    H = np.fft.fft(h, 1024)        # channel IR zero-padded to 1024
    X = Y / H                       # equalise all bins
    data_bins = X[1:511]
    return data_bins

def block_symbolise(equilised_data):
    b_list = []

    #Assign symbols to bits according to quadrant of constellation point
    for symbol in equilised_data:
        if symbol.real > 0 and symbol.imag > 0:
            b_list.append('0')
            b_list.append('0')
        elif symbol.real < 0 and symbol.imag > 0:
            b_list.append('0')
            b_list.append('1')
        elif symbol.real < 0 and symbol.imag < 0:
            b_list.append('1')
            b_list.append('1')
        else:
            b_list.append('1')
            b_list.append('0')
    return b_list

#print(split_pattern(samples)[0])

blocks = split_pattern(samples)
master_bit_list = []

for i in range(0, len(blocks), 2):
    prefix = blocks[i]
    data_block = blocks[i+1]

    equalised_data = channel_equalise(data_block, channelResponse)
    if i == 4:
        print(np.shape(equalised_data))
        print(equalised_data[0:10])
        plot_argand_complex(equalised_data)

    #Complete Gray's encoding
    block_bit_list = block_symbolise(equalised_data)
    
    if i == 2:
        print("Block 2 bit list:")
        print(np.shape(block_bit_list))
        print(block_bit_list[0:10])
    master_bit_list.extend(block_bit_list)

print(np.shape(master_bit_list))
print(f"Master bit list {master_bit_list[0:1000]}")