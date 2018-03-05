import os
import time
from colored import fg, bg, attr

def file_lines(thefilepath):
    count = 0
    thefilepath = os.path.expanduser(thefilepath) # let the python know what is the meaning of ~/

    #with open( thefilepath, 'r' as 'r' ):
    #    for cnt ,line in enumerate(f):
    #        pass
    #count = cnt + 1
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count( b'\n' )
    thefile.close( )
    return count

def load_class_names(class_names_proxy):
    if isinstance(class_names_proxy, str) :
        class_names_proxy = os.path.expanduser(class_names_proxy)
        with open(class_names_proxy, 'r') as fp:
            lines = fp.readlines()
        return [ line.rstrip() for line in lines ]
    elif isinstance(class_names_proxy, list):
        return class_names_proxy
    else:
        raise ValueError("unknown class_names type: {}".format(str(type(class_names_proxy))))

def logging(message, color='green'):
    msg = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
    colored_msg = '%s %s %s'%(fg(color), msg, attr(0))
    print(colored_msg)

def warning(message):
    colored_msg = '%s %s %s'%(fg('red'), message, attr(0))
    print(colored_msg)
