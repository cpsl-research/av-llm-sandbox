set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash

# Make the folder to save to
mkdir -p $DATAFOLDER

# Download the kitti dataset
./../submodules/avstack-api/data/download_KITTI_ImageSets.sh $DATAFOLDER
./../submodules/avstack-api/data/download_KITTI_object_data.sh $DATAFOLDER
./../submodules/avstack-api/data/download_KITTI_raw_data.sh $DATAFOLDER
./../submodules/avstack-api/data/download_KITTI_raw_tracklets.sh $DATAFOLDER