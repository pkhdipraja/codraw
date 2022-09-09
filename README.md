# codraw

This is an implementation of neural drawer model from [CoDraw](https://github.com/facebookresearch/codraw-models). As the initial aim is to modernize the source code and reproduce the results, I only make minimal changes wherever necessary due to time constraint. The drawer model in `model/model.py` and its corresponding helper in `model/model_utils.py` can be merged into one class if no further extension is desired. The dataloader for dev and test set will probably need some work, as it is intrinsically tied to the evaluation function. The dataloader function for training is already modernized with `torch.utils.data.Dataset`.

## Training
You can set the path to CoDraw JSON file in `cfgs/path_cfgs.py`. A sample config file to train the model in the paper already exists (`cfgs/sample_config.yml`). The CoDraw dataset repository is expected to be at the same level with this repository, similar to the original configuration. The following script will run training:

```bash
python3 main.py --RUN train --MODEL_CONFIG <model_config>
```
with checkpoint saved under `/ckpts`.

Important parameters:
1. `--VERSION str`, to assign a name for the model.
2. `--GPU str`, to train the model on specified GPU. For multi-GPU training, use e.g. `--GPU '0, 1, 2, ...'`.
3. `--SEED int`, set seed for this experiment.
4. `--NW int`, to accelerate data loading speed.

To check all possible parameters, use `--help`. Currently `--RESUME` and `--SPLIT` flags can be ignored.

## Testing
You can evaluate on validation or test set using `--RUN {val, test}`. For example:
```bash
$ python3 main.py --RUN test --MODEL_CONFIG <model_config> --CKPT_V <model_version> --CKPT_E <model_epoch>
```
or with absolute path:
```bash
$ python3 main.py --RUN test --MODEL_CONFIG <model_config> --CKPT_PATH <path_to_checkpoint>.pkl
```

The seed to reproduce the final result in the paper is 38957599 (for neural drawer evaluations against script, in which they use split 'b' in the official implementation).
