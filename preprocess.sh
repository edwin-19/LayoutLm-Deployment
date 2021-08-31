wget https://guillaumejaume.github.io/FUNSD/dataset.zip

unzip dataset.zip && mv dataset data && rm -rf dataset.zip __MACOSX

mkdir data/infer_data/

python preprocess.py --data_dir data/training_data/annotations/ --data_split train --output_dir data/infer_data/ --model_name_or_path bert-base-uncased --max_len 510

python preprocess.py --data_dir data/testing_data/annotations/ --data_split test --output_dir data/infer_data/ --model_name_or_path bert-base-uncased --max_len 510

cat data/infer_data/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > data/infer_data/labels.txt