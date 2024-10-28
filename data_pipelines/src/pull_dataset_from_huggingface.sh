echo "Script Inputs: $*"


pip install datasets 
pip install ijson
pip install huggingface_hub

python pull_dataset_from_huggingface.py $*