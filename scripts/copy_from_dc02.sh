# Script to copy the compiled image file from a container running on DC02 to the local machine
# and extracts it ready for flasing to an uSD card.
# date 02/14/202
# author: Ron

#!/bin/bash

# Remote host details
REMOTE_HOST="dc02.local"
REMOTE_USER="reggler"

# Docker container ID
#CONTAINER_ID="reggler-rzv2l_vpl_v3.0.0_develop"
CONTAINER_ID="reggler-rzv2l_vlp_v3.0.0_develop"

# File to copy
#SOURCE_DIR="/home/yocto/rzv_vlp_v3.0.0/build/tmp/deploy/images/smarc-rzv2l"
SOURCE_DIR="/home/yocto/rzv_vlp_v3.0.0/build/tmp/deploy/images/smarc-rzv2l"

BZIPFILENAME="mistysom-image-smarc-rzv2l.wic.bz2"
IMGFILENAME="mistysom-image-smarc-rzv2l.wic"

REM_DST_DIR="/tmp/tempdir"

# Local directory to copy the file to
LOCAL_DIR="."

# Command to execute on the remote host
#CMD="docker cp $CONTAINER_ID:$SOURCE_DIR /tmp/tempdir"
CMD="docker cp ${CONTAINER_ID}:${SOURCE_DIR} ${REM_DST_DIR}"

# SSH into the remote host, execute the command, and then scp the file from the remote /tmp to the local directory
ssh -t $REMOTE_USER@$REMOTE_HOST "$CMD"

scp ${REMOTE_USER}@${REMOTE_HOST}:${REM_DST_DIR}/${BZIPFILENAME} ${LOCAL_DIR}
# Extract the bz2 file from the tempfile
pv ${BZIPFILENAME} | bunzip2 > ${IMGFILENAME}

# Clean up the temporary file
#rm $LOCAL_DIR/tempfile
ssh -t $REMOTE_USER@$REMOTE_HOST "rm -r ${REM_DST_DIR}"
rm ${BZIPFILENAME}


echo "File copied and extracted successfully."
