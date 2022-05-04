#
# call:
#
# ./run.sh train|predict|prep
#

BASE_PATH="."

export CUDA_VISIBLE_DEVICES=2

NAME_PATTERN="*/*.tif"

PROJECT=fibrosen_bi,folder


IMG_PATH=$BASE_PATH/samples/samples-rsir-raw.pickle
MODEL_PATH=$BASE_PATH/models/model-fibrosen-recut-raw-bi-x24.h5
OUT_PATH=rsir-x24.csv


if [ -z "$1"  ]; then
    echo "Missing arguments"
    exit
fi

ACTION=$1

echo python code/classifier_main.py $PROJECT "$MODEL_PATH" "$IMG_PATH" "$NAME_PATTERN" "$OUT_PATH" $ACTION
python code/classifier_main.py $PROJECT "$MODEL_PATH" "$IMG_PATH" "$NAME_PATTERN" "$OUT_PATH" $ACTION

echo "COMPLETED: $PROJECT "$MODEL_PATH" "$IMG_PATH" "$NAME_PATTERN" "$OUT_PATH" $ACTION"

