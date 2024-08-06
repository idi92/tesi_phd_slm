
class PureIntegrator:

    def __init__(self):
        self.reset()
        self._gain = -0.5

    def process_delta_command(self, delta_command):
        self._last_delta_command = delta_command
        self._command += delta_command * self._gain
        return self._command

    def command(self):
        return self._command

    def reset(self):
        self._command = 0
