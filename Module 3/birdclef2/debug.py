import numpy as np


class Debugger:
    def __init__(self) -> None:
        self.enabled = False
        self.weights = None
        self.x = []
        self.mel = []
        self.l = []

    def enable(self):
        self.enabled = True

    def set_attention_weights(self, weights):
        # self.weights = weights.detach().cpu().numpy()
        pass

    def record_training_batch(self, x, mel, l=None):
        if not self.enabled:
            return

        x = x.detach().cpu().numpy()
        mel = mel.detach().cpu().numpy()
        self.x.append(x)
        self.mel.append(mel)
        if l is not None:
            self.l.extend(list(l))

    def reset(self):
        self.weights = None
        self.x.clear()
        self.mel.clear()

    def on_exit(self):
        if not self.enabled:
            return
        x = np.concat(self.x, axis=0)
        mel = np.concat(self.mel, axis=0)
        np.savez("x.npz", x=x, mel=mel, l=self.l)


debugger = Debugger()
