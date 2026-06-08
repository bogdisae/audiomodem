import matplotlib.pyplot as plt
import numpy as np
import os


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


def save_Unicode_text(data_bytes, length, filename):
    print(f"data_bytes length: {len(data_bytes)}, length param: {length}")
    unicode_string = ''.join(chr(b) for b in data_bytes[:length])
    print(f"unicode_string length: {len(unicode_string)}")

    f_name = filename.split('/')[-1]
    full_path = os.path.abspath('Main Pipeline Final/Received Files/' + f_name)
    print(f"Writing to: {full_path}")

    os.makedirs('Main Pipeline Final/Received Files', exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(unicode_string)

def save_wav_file(data_bytes, length, filename):

    # Keep only the actual transmitted payload
    wav_bytes = bytes(data_bytes[:length])

    # Clean filename
    f_name = filename.split('/')[-1]

    # Ensure extension
    if not f_name.endswith('.wav'):
        f_name += '.wav'

    output_name = f_name.replace('.wav', '_Output.wav')

    # Write raw WAV bytes directly
    os.makedirs('Main Pipeline Final/Received Files', exist_ok=True)
    with open('Main Pipeline Final/Received Files/' + output_name, 'wb') as f:
        f.write(wav_bytes)

    print(f"Saved WAV file: {output_name}")
    print(f"Bytes written: {len(wav_bytes)}")
