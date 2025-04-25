set -e

MODELFOLDER=${1:-/data/$(whoami)/models}
MODELFOLDER=${MODELFOLDER%/}  # remove trailing slash

mkdir -p $MODELFOLDER
./submodules/avstack-core/models/download_mmdet_models.sh $MODELFOLDER
./submodules/avstack-core/models/download_mmdet3d_models.sh $MODELFOLDER