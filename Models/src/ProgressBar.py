import numpy as np
import sys
import os
import warnings


def clear_last_line():
    """
    Standard ANSI utility to shift the cursor up and wipe the previous line.
    """
    # \033[F  moves the cursor to the beginning of the previous line
    # \033[K  clears from the cursor to the end of the line
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.flush()


def _warmup_gpu():
    """
    Primes the NVIDIA Blackwell JIT cache by running a dummy training step.
    This forces driver/kernel logs to dump before the ProgressBar starts to
    ensure the terminal UI stays clean.
    """
    import tensorflow as tf

    print("Initializing Blackwell Kernels...", end="", flush=True)

    # dummy model to trigger JIT comp warning
    dummy = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2, activation='softplus')  # Mean/Var output
    ])
    dummy.compile(optimizer='adam', loss='mse')

    # Run one fake training step to trigger the driver logs
    dummy.fit(np.zeros((1, 10)), np.zeros((1, 2)), epochs=1, verbose=0)
    print(" Done.", flush=True)


def hide_warnings():
    """Might have to call before importing TF"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 hides warnings for cleaner JIT logs
    warnings.filterwarnings("ignore")
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')


class ProgressBar:
    """
    Terminal UI for tracking training progress.
    """
    BLANK = "░"
    FULL = "█"

    def __init__(self, total_bars: int = 30, warmup_gpu=True):
        self.total_bars = total_bars
        self.current_bar = 0
        self.total_steps = 0
        self.ratio = 0
        self.max_steps = 0  # for tracking reset and setting ratio

        # keep annoying messages out of the progress bar
        hide_warnings()
        if warmup_gpu:
            _warmup_gpu()

    def set_max_steps(self, max_steps: int) -> None:
        """
        Defines the denominator for progress math (e.g., n_estimators or N grid search iterations).
        Immediately triggers the first draw of the empty bar.
        """
        self.max_steps = max_steps
        self.ratio = self.total_bars / max_steps
        self._display()

    def update(self, update: int) -> None:
        """
        Increments steps and calculates if the bar needs to be redrawn.
        Triggers a reset and moves to a new line once 100% is reached.
        """
        if not self.ratio:
            # Orange warning if the team forgot to set the limit
            print("\033[38;5;208mMust set max steps before updating!\033[0m")
            return

        # calculate progress bar update
        self.total_steps += update
        last = self.current_bar
        self.current_bar = np.ceil(self.total_steps * self.ratio)

        # redraws progress bar if it has changed
        if last != self.current_bar:
            self._display()

        # reset
        if self.total_steps >= self.max_steps:
            print("")  # Preserve the 100% bar on screen
            self.reset()

    def reset(self):
        """Resets counters for the next training run."""
        self.total_steps = 0
        self.current_bar = 0

    def _display(self):
        """Internal refresh of the ANSI carriage return (\r) display line."""
        bar_chars = [self.FULL if i <= self.current_bar else self.BLANK
                     for i in range(self.total_bars)]
        bar_str = "".join(bar_chars)
        print(f"\r{bar_str}", end="", flush=True)