for topic in `rostopic list -b subset.bag` ; do rostopic echo -p -b subset.bag $topic >bagfile-${topic//\//_}.csv ; done
