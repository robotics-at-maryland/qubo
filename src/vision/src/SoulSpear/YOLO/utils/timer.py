from time import time
import numpy as np

class Timer():
    def __init__(self):
        self.record = {}
        self.timeline = {}
        self.id = 0
        self.max = 10
        self.tictac = None

    def terminate(self):
        if self.id == self.max : return True
        else : False

    def event(self,description):
        if self.terminate(): return None
        if self.id in self.record:
            self.record[ self.id ]['d'].append(description)
            self.record[ self.id ]['t'].append(time())
        else:
            self.record[self.id] = {'d' :[description],'t': [time()] }
    def release(self):
        self.record = {}

    def calculate(self):
        if self.terminate(): return None

        record = self.record[self.id]  # copy

        record_d = np.array(record['d'][1:])
        record_t = np.array(record['t'])

        timeline = ( record_d, record_t[1:] - record_t[0:-1] )

        self.timeline[self.id] = timeline

    def report( self, mode = "avg" ,release = True ):
        if self.terminate(): return None

        if mode.lower() == "rec":
            for d, t in zip( *self.timeline[self.id] ):
                print( "{:<20} : {:.3f}s".format(d,t ) )

        elif mode.lower() == "avg":
            all_timeline = np.array( self.timeline.values() )
            timeline_t = timeline[:,:,1]
            timeline_d = timeline[:,:,0]

            assert np.nunique(timeline_d) == len( timeline_d[0] ) , "your description is not consistant"
            timeline = ( timeline_d[0] , np.mean(timeline_t) )
            for d, t in zip( timeline ):
                print( "{:<15} : {}s".format(d,int(t)) )
            print( "{:<15} : {}s".format('total', sum(timeline) ) )

        if release :
            self.release()
        else :
            self.id += 1

    def tic(self):
        self.tictac = time()

    def tac(self):
        recent = time()
        interval = recent - self.tictac
        self.tictac = recent
        return interval
