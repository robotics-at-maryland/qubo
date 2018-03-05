import os
import torch
import shutil
import warnings

checkpoint_log = "Can't find the checkpoint file: {}"

def check_file( path, log_format  ):
    exist = os.path.isfile( path )
    if not exist:
        print( log_format.format( path ) )
        return False
    else :
        return True

def save_checkpoint( portfolio, filename ='../checkpoint/UN_checkpoint.pth.tar',bestfilename='../checkpoint/UN_best.pth.tar', is_best=False):
    filename = os.path.expanduser( filename )
    bestfilename = os.path.expanduser( bestfilename )
    maybe_folder( os.path.dirname(filename) )
    print('saving model to {}'.format( os.path.abspath(filename) ) )
    torch.save( portfolio, filename )
    if is_best:
        print('copy the checkpoint to the best')
        shutil.copyfile( filename, bestfilename )

def load_checkpoint( model, optimizer,filename ='../checkpoint/UN_checkpoint.pth.tar',bestfilename='../checkpoint/UN_best.pth.tar', which = 1):
    # in reinforcement learning, the episode is not actually matter
    # we just recover the model and optimizer from blank
    # steps_done is also required to continues resume epislon Greedy policy

    global GENERATION
    global GLOBAL_STEP

    filename = os.path.expanduser( filename )
    bestfilename = os.path.expanduser( bestfilename )
    filename = filename if which == 1 else bestfilename
    maybe_folder( os.path.dirname(filename) )

    if check_file( filename, checkpoint_log ):
        print( "===> Loading checkpoint from {}".format( os.path.abspath(filename) ) )
        checkpoint = torch.load(filename)

        GENERATION = checkpoint['GENERATION'] + 1
        print( 'The generation of this model is :', GENERATION )

        GLOBAL_STEP = checkpoint['GLOBAL_STEP']
        print( 'The global step is :', GLOBAL_STEP )

        model.load_state_dict(checkpoint['model'])

        if not checkpoint['optimizer'] == None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        #if not arch == checkpoint['arch']:
        #     warnings.warn( 'the architecture info in the checkpoint=>>> {} does not match the parameters=>>> {}'.format( checkpoint['arch'],arch ) ) #, Warning )

def load_model(model,filename ='../checkpoint/UN_checkpoint.pth.tar',bestfilename='../checkpoint/UN_best.pth.tar', which = 1):
    filename = filename if which == 1 else bestfilename
    filename = os.path.expanduser( filename )
    if check_file( filename, checkpoint_log ):
        print('"===>load model from {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict( checkpoint['model'] )

        #if not arch == checkpoint['arch']:
        #    warnings.warn('the architecture info in the checkpoint=>>> {} does not match the parameters=>>> {}'.format( checkpoint['arch'],arch ) ) #, Warning)

def erase_checkpoint( filename ='../checkpoint/UN_checkpoint.pth.tar' ):
    filename = filename.format( arch )
    if os.path.isfile( filename ):
        os.remove( filename )
        print( 'erase checkpoint from: ', os.path.abspath( filename ) )

def maybe_folder(path):
    os.makedirs(path) if not os.path.isdir( path ) else None
    #_ = os.mkdir(path) if not os.path.isdir( path ) else None

def create_portfolio(model,optimizer,generation,global_step,arch):
        portfolio={
            'GENERATION': generation,
            'GLOBAL_STEP': global_step,
            'arch': arch, # architecture
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
        return portfolio
