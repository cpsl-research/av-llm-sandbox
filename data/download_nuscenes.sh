set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash

DATASET=${2:-mini}

# Make the folder to save to
mkdir -p $DATAFOLDER

# Download the nuScenes data
if [ "$DATASET" -eq "mini" ]; then
    ./submodules/avstack-api/data/download_nuScenes_mini.sh $DATAFOLDER
    ./submodules/avstack-api/data/download_nuScenes_CAN.sh $DATAFOLDER
elif [ "$DATASET" -e "full" ]; then
    echo "ERROR: For now, you have to download the full nuScenes dataset manually!"
    exit 1
else 
    echo "ERROR: Did not pass an acceptable version of nuScenes dataset! Options are [mini, full]"
    exit 1
fi

echo "Done!"