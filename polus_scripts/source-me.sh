if [ -z "$1" ]
  then
    echo "You must supply experiment name in argument!"
fi
conda activate pytorch
export TRAINHOME=$CONDA/raft
export OUTPUTS="$TRAINHOME/runs/$1"