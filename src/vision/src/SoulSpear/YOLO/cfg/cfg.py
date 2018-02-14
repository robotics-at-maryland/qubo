import json
import os

cfg_root = os.path.dirname(__file__)

def parse_cfg( name ):
    cfg_file = os.path.join( cfg_root, "{}.json".format( name ) )
    print( "load config from : ", os.path.abspath(cfg_file) )
    with open( cfg_file , 'r' ) as f:
        cfg = json.load( f )
    return cfg

def modify_cfg( name, cfg ):
    cfg_file = os.path.join( cfg_root, "{}.json".format( name ) )
    cfg = json.dumps( cfg , indent = 4 )
    with open( cfg_file , 'w') as f:
        f.write( cfg )
