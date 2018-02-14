import numpy as np

class Logger():
    def __init__(self, target='loss'):
        self.record = {}
        self.epoch = []
        self.is_best = False
        self.best = np.inf
        self.target = target

    def log(self, header, value):
        if header in self.record:
            self.record[header].add(value)
        else:
            self.record[header] = set((value,))

    def epoch_log( self ):
        self.is_best = False
        self.epoch.append(dict())
        for key, value in self.record.items():
            value = list(value) # np can't operate "set" object
            statistic = {
                'max':np.max(value),
                'min':np.min(value),
                'mean':np.mean(value),
                'median':np.median(value),
                }

            self.epoch[-1][key] = statistic
        self.record = {}

        self.epoch_loss = self.epoch[-1]['loss']['mean']
        if self.epoch_loss < self.best :
            self.best = self.epoch_loss
            self.is_best = True
