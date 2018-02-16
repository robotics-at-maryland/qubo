from models.tiny_yolo import TinyYoloNet
from transfer.basic import load_conv_bn
from utils.model_io import portfolio, save_checkpoint, maybe_folder
from cfg.cfg import load_conv_bn

def load_weights( model, path, save_last=True ):
    #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
    buf = np.fromfile(path, dtype=np.float32)
    start = 4

    start = load_conv_bn( buf, start, model.cnn[0], model.cnn[1] )
    start = load_conv_bn( buf, start, model.cnn[4], model.cnn[5] )
    start = load_conv_bn( buf, start, model.cnn[8], model.cnn[9] )
    start = load_conv_bn( buf, start, model.cnn[12], model.cnn[13] )
    start = load_conv_bn( buf, start, model.cnn[16], model.cnn[17] )
    start = load_conv_bn( buf, start, model.cnn[20], model.cnn[21] )

    start = load_conv_bn( buf, start, model.cnn[24], model.cnn[25] )
    start = load_conv_bn( buf, start, model.cnn[27], model.cnn[28] )

    if save_last:
        start = load_conv( buf, start, model.cnn[30] )

if __name__ == '__main__':
    cfg = parse_cfg( 'TinyYoloNet' )
    model = TinyYoloNet( cfg.model['anchors'], cfg.model['num_classes'] )
    load_weights( model, cfg.trans['path'], cfg.trans['save_last'] )

    maybe_folder( cfg.checkpoint.root )
    portfolio = create_portfolio( model=model ,optimizer=None ,generation=0, global_step=0, arch='TinyYoloNet' )
    save_checkpoint( portfolio , is_best=False, cfg.checkpoint.path, cfg.checkpoint.best )
