import os
import sys
import random
import argparse
from pathlib import Path

import skvideo.io
import skimage.io

__all__ = ["time_int_ext", "nevigate", "inspect"]

parser = argparse.ArgumentParser(
    description='Find out all the .avi file in the directoryand extract frame from those videos')
parser.add_argument('--mode', '-m',
    description="test 4 debug, run 4 extract, ins 4 inspectation",
    default='run',type= str
    )
parser.add_argument(
    '--idir',
    '-i',
    description="root directory for video files",
    type=str
    )
parser.add_argument(
    '--odir',
    '-o',
    description="root directory for frame files",
    type=str
    )

def time_int_ext( path, idir, odir, frame_int ):
    """Read the video file from a given path and save some frames of it

    Arguments:
        path (:class:'Path' or str): The path of the video file
        idir (:class:'Path' or str): Root folder of the video file
        odir (:class:'Path' or str): Root folder of the frame file.
            It would be created if it is not exist.
        frame_int (int): The number of the frame per saving frame
    """

    # video reading header
    videogen = skvideo.io.vreader( str( path ) )

    frame_loop = 0
    frame_num = 0

    # read video frame by frame and save it to png
    for frame in videogen:
        # pick a frame in every "frame_int" interval
        if frame_loop == 0:
            frame_pick = random.randint( 0 , frame_int )
        if frame_pick == frame_loop:
            try:
                _save( frame, _swap_root( path, idir, odir) ,frame_num )

            except Exception as e:
                sys.stderr.write( 'err: {}'.format( e ) )
                sys.stderr.flush()
                print( frame.shape )
                print( frame.dtype )
                print("can't save frame: ", frame_num )

        frame_num += 1
        frame_loop = ( frame_loop + 1 )%frame_int

def _swap_root( path, idir, odir):

    return Path( str(path).replace( idir , odir ) )

def _scroll( text ):
    sys.stdout.write( "\r>>> {:<80} <<<".format( text ) )
    sys.stdout.flush()

_skimsave = skimage.io.imsave

def _save( frame, path, frame_num ):
    path.suffix = ''  # erase the suffix
    path = str( path )

    os.makedirs( path, exist_ok = True ) # create the directory
    path = os.path.join( path, '{}.png'.format(frame_num) )
    _skimsave( path, frame ) # save the file in png format
    _scroll( path ) # show the path on the terminal

def nevigate( folder, pattern ):
    root = Path( folder )
    return root.rglob( pattern )

# home-make version

#def nevigate( folder, pathlist ):
#    root = Path( folder )
    #print( root )
#    folders = [ x for x in root.iterdir() if x.is_dir() ]
#    files = [ x for x in root.iterdir() if x.is_file() ]
#    video = [ x for x in files if x.suffix == ".avi" ]
#    pathlist = pathlist + video

#    for f in folders:
#        pathlist = nevigate( f, pathlist )
#    return pathlist


def inspect( im_files, frame_rate ):
    """Open a plt window to inspect the generated frames

    Arguments:
        im_files (generator or arraylike): Path info of the
            frame png file.
        frame_rate (int): Desired frame rate for display.
    """
    def worker( ID , im_files , q ):
        # load image from the file and push it to the queue
        for f in im_files:
            frame = skimage.io.imread( f )
            q.put( frame )
            _scroll( str(f) )
        q.put(None)

    # set up
    im_files = [ f for f in im_files ]
    length = len(im_files)
    ncore = 4
    finish = 0

    load_queue = mp.Queue( maxsize = 100 )
    seg = length//ncore
    img_workers = list()

    # initialize the workers
    for i in range( ncore ) :
        args = (i , im_files[ i*seg : (i+1)*seg ] , load_queue )
        worker = mp.Process( target = worker, args = args )
        img_workers.append( worker )

    for img_worker in img_workers: img_worker.start()

    # initialize the plt window
    fig, ax = plt.subplots( 1,1, figsize = ( 6, 4.5 ) )
    plt.axis( 'off' )
    plt.tight_layout()
    frame = load_queue.get()
    win = ax.imshow( frame )

    # real time ploting the image
    while True:
        frame = load_queue.get()

        # run until all worker is stopped
        if frame is None :
            finish += 1
            if finish<ncore: continue
            else : break

        # plot
        win.set_data( frame )
        plt.pause( 1/frame_rate )

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode.lower() == 'run':

        video_list = nevigate(args.idir, '*.avi')
        video_list = filter(lambda x: not "error" in str(x), video_list)

        for video in video_list:
            time_int_ext(video, args.idir, args.odir , 50)

    elif args.mode.lower() == 'ins':

        import matplotlib.pyplot as plt
        import time
        import multiprocessing as mp

        plt.ion()

        frame_gen = nevigate( args.idir, '*.png' )
        inspect( frame_gen, 20 )

    elif args.mode.lower() == 'test':
        import skvideo.datasets

        plist = []
        plist = nevigate("/home/aurora/video", '*.avi')
        print( next( plist ) )
        videogen = skvideo.io.vreader(skvideo.datasets.bigbuckbunny())

        for i , frame in enumerate( videogen ):
            if i%30 == 0: print( frame.shape )

        skimage.io.imshow( frame )

    else:
        sys.stderr.write("three mode : run, ins, and test \n")
        sys.stderr.flush()

    print()
