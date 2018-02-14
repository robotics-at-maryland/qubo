import skvideo.io
import skvideo.datasets
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser( description = '....' )
parser.add_argument( '--test','-t', default=False,type= bool )

if __name__ == "__main__":
    args = parser.parse_args()

    if args.test == True:
        import time
        import matplotlib.pyplot as plt
        inputparameters = {}
        outputparameters = {}
        videogen = skvideo.io.vreader( skvideo.datasets.bigbuckbunny())
        #pbar = tqdm(total = len( videogen ) )
        i = 0
        for frame in videogen :
            if i%30 == 0: print( frame.shape )
            i += 1

        plt.imshow( frame )
        plt.axis("off")
        plt.show()
        time.sleep(10)
