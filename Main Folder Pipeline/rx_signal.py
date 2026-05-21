class RxSignal:
    def __init__(self, sigArray):
        self.sigArray = sigArray # The actual signal array (numpy)
        self.dataIdx = None # Represents when we think the DATA starts (NOT the chirp)

        # Define the range of samples where we think the key will be
        # E.g if we think key is from sample 1000->5000, start = 1000, end = 5000. Used for channel estimation
        self.estimationIdxStart = None
        self.estimationIdxEnd = None