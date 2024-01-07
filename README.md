# Decomposable Transformer Point Process (DTPP) model
This repository contains the Pytorch implementation of the DTPP method from the paper [Decomposable Transformer Point Process (DTPP) model](https://arxiv.org).

## Environment Requirements

Ensure that all the dependencies have been installed using the `requirements.txt` file.

## Data preparation
All datasets can be downoladed from [Google-Drive-1](https://drive.google.com/drive/folders/13e5jCkprJGB6jiVtIrU-XaCzSws5PPfB) and [Google-Drive-2](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w). The datasets are already preprocessed and ready to be used. Unzip the files (train.pkl, dev.pkl, test.pkl) and put them in the [./data/{dataset_name}](https://github.com/aresPanos/dtpp/tree/main/data) directory.

## Train DTPP model on real-world datasets and Evaluate it on next-event prediction

We train and evaluate the inter-event time model and the mark model, separately. 

To train and evaluate the time model using two mixture components on Taxi dataset, run

    python code/run/train_eval_time_dist.py --dataset taxi --data_dir <directory_of_data> --log_dir <directory_of_logs>

The learned parameters of the mixture of log-Normals are stored at  `<directory_of_logs>/taxi/time_dist/saved_models/model_numMixtures-2.pt`

Next, to train and evaluate the mark model using D=128 and L=2 layers for the Transformer architecture, run

    python code/run/train_eval_mark_dist.py --dataset taxi -dmodel 64 -nLayers 2 --data_dir <same_as_above> --log_dir <same_as_above>

The learned parameters of the model are stored at  `<directory_of_logs>/taxi/mark_dist/saved_models/XFMRNHPFast_dmodel-64_nLayers-2_nHeads-2_date-time.pt`

## Long-horizon prediction

To perform long-horizon prediction, we only need to define the directories of the two trained models from the previous step, together with the architecture of the Transformer,

    python code/run/multi_step_eval.py --dataset taxi -dmodel 64 -nLayers 2 --data_dir <same_as_above> --log_dir <same_as_above> --model_times_dir <saved_model_of_times> --model_marks_dir <saved_model_of_marks>

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/aresPanos/dtpp/blob/main/LICENSE) file for details.

## Acknowledgements
Our code is based on the following repositories:
* [Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)
* [HYPRO](https://github.com/ant-research/hypro_tpp/tree/main)

If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

## Citation
If you use the code for any published work, please cire the following paper:
@inproceedings{,
  author = {},
  title = {},
  booktitle = {},
  year = {2024}
}
