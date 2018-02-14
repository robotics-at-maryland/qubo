from models.tiny_yolo import TinyYoloNet
from transfer.basic import load_conv_bn
from utils.model_io import portfolio, save_checkpoint, maybe_folder
from cfg.cfg import load_conv_bn

def load_weights( model, path, save_last=True ):
    #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
    buf = np.fromfile(path, dtype=np.float32)
    start = 4

    start = load_conv_bn( buf, start, model.cnn[0], self.cnn[1] )
    start = load_conv_bn( buf, start, model.cnn[4], self.cnn[5] )
    start = load_conv_bn( buf, start, model.cnn[8], self.cnn[9] )
    start = load_conv_bn( buf, start, model.cnn[12], self.cnn[13] )
    start = load_conv_bn( buf, start, model.cnn[16], self.cnn[17] )
    start = load_conv_bn( buf, start, model.cnn[20], self.cnn[21] )

    start = load_conv_bn( buf, start, model.cnn[24], self.cnn[25] )
    start = load_conv_bn( buf, start, model.cnn[27], self.cnn[28] )

    if save_last:
        start = load_conv( buf, start, model.cnn[30] )

if __name__ == '__main__':
    cfg = parse_cfg( 'TinyYoloNet' )
    model = TinyYoloNet( cfg.model['anchors'], cfg.model['num_classes'] )
    load_weights( model, cfg.trans['path'], cfg.trans['save_last'] )

    maybe_folder( cfg.checkpoint.root )
    portfolio = create_portfolio( model=model ,optimizer=None ,generation=0, global_step=0, arch='TinyYoloNet' )
    save_checkpoint( portfolio , is_best=False, cfg.checkpoint.path, cfg.checkpoint.best )
