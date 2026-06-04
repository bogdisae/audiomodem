import matplotlib.pyplot as plt
import numpy as np

def repeated_chirp(length, repeats,silence):
    signal = np.linspace(0, np.pi, length)  # simple linear chirp from 0 to π
    # Apply silence before and after the chirp
    section = np.concatenate([silence, signal])
    signal = np.concatenate([section for _ in range(repeats)])
    signal = np.concatenate([signal, silence])  # Add silence at the end
    m = np.max(np.abs(signal))
    return signal / m if m != 0 else signal

def chirp_chirp_inv(length, repeats, silence):
    signal = np.linspace(0, np.pi, length)  # simple linear chirp from 0 to π
    # Apply silence before and after the chirp
    section = np.concatenate([silence, signal, silence, signal[::-1]])
    signal = np.concatenate([section for _ in range(repeats)])
    signal = np.concatenate([signal, silence])
    m = np.max(np.abs(signal))
    return signal / m if m != 0 else signal

def DBPSK_chirp(length, repeats, silence, pattern):
    signal = np.linspace(0, np.pi, length)  # simple linear chirp from 0 to π
    # Apply silence before and after the chirp
    input_pattern = [signal * i for i in pattern]  # Modulate the chirp by the pattern
    section = np.concatenate([piece for chirp in input_pattern for piece in (silence, chirp)])
    signal = np.concatenate([section for _ in range(repeats)])
    signal = np.concatenate([signal, silence])
    m = np.max(np.abs(signal))
    return signal / m if m != 0 else signal

silence = np.zeros(512)
r_chirp = repeated_chirp(1024, 4, silence)
inv_chirp = chirp_chirp_inv(1024, 2, silence)
dbpsk_chirp = DBPSK_chirp(1024, 1, silence, [1, -1, 1, -1])


fig, axes = plt.subplots(4, 1, figsize=(5, 5))
ax1, ax2, ax3, ax4 = axes.flatten()
ax1.plot(r_chirp)
ax1.set_title("Repeated Chirp")
ax1.set_ylabel("Frequency")
ax2.plot(inv_chirp)
ax2.set_title("Bi-Chirp")
ax2.set_ylabel("Frequency")
ax3.plot(np.abs(dbpsk_chirp))
ax3.set_title("PCBPSK Chirp")
ax3.set_ylabel("Frequency")
ax3.set_xlabel("Sample Index")
phase = np.angle(dbpsk_chirp)
ax4.plot(phase)
#set y axis to be between -pi and pi
ax4.set_ylim(-1, 4)
ax4.set_title("Phase of PCBPSK Chirp")
ax4.set_xlabel("Sample Index")
ax4.set_ylabel("Phase")
ax4.autoscale(False)
plt.tight_layout()
plt.show()

def correlate(signal, key, mode='full'):
    return np.correlate(signal, key, mode=mode)


fig, axes = plt.subplots(3, 1, figsize=(5, 5))
ax1, ax2, ax3 = axes.flatten()
corr_r_chirp = correlate(r_chirp, r_chirp)
ax1.plot(corr_r_chirp)
ax1.set_title("Autocorrelation of Repeated Chirp")
corr_inv_chirp = correlate(inv_chirp, inv_chirp)
ax2.plot(corr_inv_chirp)
ax2.set_title("Autocorrelation of Bi-Chirp")
corr_dbpsk_chirp = correlate(dbpsk_chirp, dbpsk_chirp)
ax3.plot(np.abs(corr_dbpsk_chirp))
ax3.set_title("Magnitude of Autocorrelation of PCBPSK Chirp")
ax3.set_xlabel("Lag")
plt.tight_layout()
plt.show()

dbpsk_chirp_1 = DBPSK_chirp(1024, 1, silence, [1, -1, -1, 1])
dbpsk_chirp_2 = DBPSK_chirp(1024, 2, silence, [1, -1, 1])

fig, axes = plt.subplots(4, 1, figsize=(5, 10))
ax1, ax2, ax3, ax4 = axes.flatten()
ax1.plot(dbpsk_chirp_1)
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Frequency")
ax2.plot(np.abs(correlate(dbpsk_chirp_1, dbpsk_chirp_1)))
ax2.set_xlabel("Lag")
ax2.set_ylabel("Correlation Magnitude")

ax3.plot(dbpsk_chirp_2)
ax3.set_xlabel("Sample Index")
ax3.set_ylabel("Frequency")
ax4.plot(np.abs(correlate(dbpsk_chirp_2, dbpsk_chirp_2)))
ax4.set_xlabel("Lag")
ax4.set_ylabel("Correlation Magnitude")

ax1.set_title("DBPSK Chirp 1")
ax2.set_title("Correlation of DBPSK Chirp 1")
ax3.set_title("DBPSK Chirp 2")
ax4.set_title("Correlation of DBPSK Chirp 2")
ax4.set_xlabel("Lag")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(5, 6))
ax1, ax2, ax3, ax4 = axes.flatten()

ax1.plot(dbpsk_chirp_1)
ax1.set_title("DBPSK Chirp 1")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Frequency")

ax2.plot(dbpsk_chirp_2)
ax2.set_title("DBPSK Chirp 2")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Frequency")

ax3.plot(np.abs(correlate(dbpsk_chirp_1, dbpsk_chirp_1)))
ax3.set_title("Correlation of DBPSK Chirp 1")
ax3.set_xlabel("Lag")
ax3.set_ylabel("Correlation Magnitude")

ax4.plot(np.abs(correlate(dbpsk_chirp_2, dbpsk_chirp_2)))
ax4.set_title("Correlation of DBPSK Chirp 2")
ax4.set_xlabel("Lag")
ax4.set_ylabel("Correlation Magnitude")

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
from scipy.signal import correlate
import numpy as np

fig = plt.figure(figsize=(6, 7))
gs = fig.add_gridspec(3, 2)

# --- Top row (spans both columns) ---
ax_top = fig.add_subplot(gs[0, :])
ax_top.plot(np.abs(dbpsk_chirp_1))  # same sequence, show once
ax_top.set_title("PCBPSK Chirp Sequence")
ax_top.set_xlabel("Sample Index")
ax_top.set_ylabel("Frequency")

# --- Middle row (Phase plots) ---
ax_phase1 = fig.add_subplot(gs[1, 0])
ax_phase1.plot(np.angle(dbpsk_chirp_1))
ax_phase1.set_title("Phase PCBPSK Chirp Sequence 1")
ax_phase1.set_xlabel("Sample Index")
ax_phase1.set_ylabel("Phase")

ax_phase2 = fig.add_subplot(gs[1, 1])
ax_phase2.plot(np.angle(dbpsk_chirp_2))
ax_phase2.set_title("Phase PCBPSK Chirp Sequence 2")
ax_phase2.set_xlabel("Sample Index")
ax_phase2.set_ylabel("Phase")

# --- Bottom row (Autocorrelation) ---
ax_corr1 = fig.add_subplot(gs[2, 0])
ax_corr1.plot(np.abs(correlate(dbpsk_chirp_1, dbpsk_chirp_1)))
ax_corr1.set_title("Autocorrelation PCBPSK Chirp Sequence 1")
ax_corr1.set_xlabel("Lag")
ax_corr1.set_ylabel("Magnitude")

ax_corr2 = fig.add_subplot(gs[2, 1])
ax_corr2.plot(np.abs(correlate(dbpsk_chirp_2, dbpsk_chirp_2)))
ax_corr2.set_title("Autocorrelation PCBPSK Chirp Sequence 2")
ax_corr2.set_xlabel("Lag")
ax_corr2.set_ylabel("Magnitude")

plt.tight_layout()
plt.show()
