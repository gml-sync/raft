if [ -z "$1" ]
  then
    echo "You must supply experiment name in argument!"
fi
conda activate pytorch
export TRAINHOME=$CONDA/$1
export OUTPUTS="$TRAINHOME/git/runs/$1"
mkdir -p $OUTPUTS