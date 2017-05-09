for topic in `rostopic list -b $1` ; do rostopic echo -p -b $1 $topic >bagfile-${topic//\//_}.csv ; done
