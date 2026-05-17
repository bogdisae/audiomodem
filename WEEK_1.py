import csv
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

channelResponse = []

# Open the CSV file
with open("WEEK_1_FILES/channel.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        # Check row exists and first column is not empty
        if row and row[0].strip() != "":
            channelResponse.append(float(row[0])) # Convert all data to floats to avoid weird errors


# Calculate DFT of channel response (for later use)
channelFFT = np.fft.fft(channelResponse, 1024)

# Open first WAV file
fs, samples = wavfile.read("WEEK_1_FILES/file01.wav")

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

def render_greyscale(data_bytes, length, width, cmap = "gray"):
    h = length // width
    data_bytes = np.array(data_bytes[:width*h]).reshape(h, width)
    
    plt.imshow(data_bytes, cmap=cmap)
    plt.show()

def render_2byte_greyscale(data_bytes, length, width, cmap = "gray"):
    arr = np.array(data_bytes[:2*(len(data_bytes)//2)], dtype=np.uint8)
    h = length // (width*2)

    data_bytes = arr.view(np.uint16).byteswap()[:width*h].reshape(h, width)

    plt.imshow(data_bytes, cmap=cmap)
    plt.show()

def render_4byte_greyscale(data_bytes, length, width, cmap = "gray"):
    arr = np.array(data_bytes[:4*(len(data_bytes)//4)], dtype=np.uint8)
    h = length // (width*4)

    data_bytes = arr.view(np.uint32).byteswap()[:width*h].reshape(h, width)

    plt.imshow(data_bytes, cmap=cmap)
    plt.show()

def render_rgb(data_bytes, length, width, offset = 0):
    data_bytes = data_bytes[offset:]
    arr = np.array(data_bytes[:3*(len(data_bytes)//3)], dtype=np.uint8)
    h = length // (width*3)

    data_bytes = arr[:width*h*3].reshape(h, width, 3)

    plt.imshow(data_bytes)
    plt.show()

def render_rgba(data_bytes, length, width, offset = 0):
    data_bytes = data_bytes[offset:]
    arr = np.array(data_bytes[:4*(len(data_bytes)//4)], dtype=np.uint8)
    h = length // (width*4)

    data_bytes = arr[:width*h*4].reshape(h, width, 4)

    plt.imshow(data_bytes)
    plt.show()

def channel_equalise(block, H):
    Y = np.fft.fft(block)
    X = Y / H # Zero-forcing
    data_bins = X[1:512]
    return data_bins

def block_symbolise(equilised_data, anticlockwise=True):
    b_list = []

    #Assign symbols to bits according to quadrant of constellation point
    for symbol in equilised_data:
        if symbol.real > 0 and symbol.imag > 0:
            b_list.append('0')
            b_list.append('0')
        elif symbol.real < 0 and symbol.imag > 0:
            if anticlockwise:
                b_list.append('0')
                b_list.append('1')
            else:
                b_list.append('1')
                b_list.append('0')
        elif symbol.real < 0 and symbol.imag < 0:
            b_list.append('1')
            b_list.append('1')
        else:
            if anticlockwise:
                b_list.append('1')
                b_list.append('0')
            else:
                b_list.append('0')
                b_list.append('1')
    return b_list


# Remove prefix and decode bits
blocks = split_pattern(samples)
master_bit_list = []

for i in range(0, len(blocks), 2):
    prefix = blocks[i]
    data_block = blocks[i+1]
    equalised_data = channel_equalise(data_block, channelFFT)
    # if i == 4:
    #     plot_argand_complex(equalised_data)
    #Complete Gray's encoding
    block_bit_list = block_symbolise(equalised_data)
    master_bit_list.extend(block_bit_list)


# Extract header information from file
data_bytes = []
for i in range (0,len(master_bit_list), 8):
    data_bytes.append(int(''.join(master_bit_list[i: i+8]), 2))

first_null = data_bytes.index(0)
second_null = data_bytes.index(0, first_null+1)
filename = ''.join(chr(b) for b in data_bytes[:first_null])
length = int(''.join(chr(b) for b in data_bytes[first_null+1:second_null]))
data_bytes = data_bytes[second_null+1:]

print(f"filename: {filename} length: {length} data start index: {second_null+1} decoded byte length: {len(data_bytes)}")

def save_Unicode_text(data_bytes, length, filename):
    unicode_string = ''.join(chr(b) for b in data_bytes[:length])

    f_name = filename.split('/')[-1]
    print(f_name)
    
    with open('WEEK_1_files/' + f_name, 'w', encoding='utf-8') as f:
        f.write(unicode_string)

# Viewing files
# File 1:
save_Unicode_text(data_bytes, length, filename)
# Image 2:
# render_greyscale(data_bytes,length, width = 400)
# Image 3: 
# render_4byte_greyscale(data_bytes, length, width = 122)
# Audio 4:

# Image 5:
# render_4byte_greyscale(data_bytes, length, width=150)
# Image 6:
# render_4byte_greyscale(data_bytes,length, width = 200) # image start misaligned
# Image 7:
# render_4byte_greyscale(data_bytes,length, width = 200)
# Image 8:
# render_4byte_greyscale(data_bytes,length, width = 200) 
# Image 9:
# render_2byte_greyscale(data_bytes,length, width = 375) 
# Image 10:
# render_4byte_greyscale(data_bytes,length, width = 200, cmap="viridis") 
# Image 11:
# render_4byte_greyscale(data_bytes,length, width = 250) 
# Image 12:
# render_rgb(data_bytes,length, width = 330, offset=9) 
# Image 13:
# render_rgba(data_bytes,length, width = 300, offset=18) 
# Audio 14:

