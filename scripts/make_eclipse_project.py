import os.path
import subprocess
import sys

def main():
    if len(sys.argv) != 2:
        print '\tUsage: python make_eclipse_project.py PROJECT_PATH'
        return
    
    bin_dir = sys.argv[1]
    if not os.path.isdir(bin_dir):
        print '\t' + bin_dir + ' is not a valid directory.'
        return
    bin_dir = os.path.abspath(bin_dir)

    #Get the path to ROS's mk package
    mk_dir = subprocess.check_output(['rospack', 'find', 'mk']).strip()
    if not os.path.isdir(mk_dir):
        print "Could not find ROS's 'mk' package."
        return

    #Get the top level directory of the source files
    script_dir = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, os.path.pardir))

    print 'Project Directory: ' + bin_dir
    print 'Source Directory: ' + src_dir

    #Run CMake in the project directory using the Eclipse generator
    subprocess.Popen(['cmake', '-G', 'Eclipse CDT4 - Unix Makefiles', src_dir], cwd = bin_dir).wait()

    #Run ROS's awk script to pass the current shell environment into the make process in Eclipse
    xml_string = subprocess.check_output(['awk', '-f', os.path.join(mk_dir, 'eclipse.awk'), os.path.join(bin_dir, '.project')])
    with open(os.path.join(bin_dir, '.project'), 'w') as f:
        f.write(xml_string)

    print 'All done.'

if __name__ == "__main__":
    main()
