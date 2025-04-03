import threading

class Timer:
    def __init__(self, interval, callback):
        """
        Initialize the timer.
        
        :param interval: The interval in seconds between function calls.
        :param callback: The function to be called at each interval.
        """
        self.interval = interval
        self.callback = callback
        self.timer_thread = None
        self.is_running = False

    def _run(self):
        """
        Internal method to run the callback and restart the timer.
        """
        self.is_running = False
        self.start()
        self.callback()

    def start(self):
        """
        Start the timer.
        """
        if not self.is_running:
            self.timer_thread = threading.Timer(self.interval, self._run)
            self.timer_thread.start()
            self.is_running = True

    def stop(self):
        """
        Stop the timer.
        """
        if self.timer_thread is not None:
            self.timer_thread.cancel()
            self.is_running = False

    def restart(self):
        self.stop()
        self.start()