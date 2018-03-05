import os

def load_model_method( ):
    import models
    model_names = sorted(name for name in models.__dict__
    if ( #name.islower()
        not name.startswith('_')
        and not name.startswith('__')
        and not 'layers' in name
        and not 'loss' in name
        and callable( models.__dict__[ name ] )
    ))
    methods = { name : models.__dict__[ name ] for name in model_names }
    return model_names, methods

def load_cfg_name():

    def rm_suffix( dirs, suffix ):
        return list( map( lambda d: d.replace(suffix,''), dirs ) )
    data_dirs = os.listdir("./cfg/data")
    arch_dirs = os.listdir("./cfg/arch")
    data_cfg = rm_suffix( data_dirs, '.json' )
    arch_cfg = rm_suffix( arch_dirs, '.json' )
    return data_cfg, arch_cfg

if __name__ == "__main__":
    model_names, methods = load_model(  )
    print( model_names )
    print( methods )
    print( load_cfg_name() )
