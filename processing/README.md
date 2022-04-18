# Structure

```
├── configs - folder with configs for the processing functions
├── legacy - outdated version of this project
├── raw_files
│   ├── of_output
├── saved - folder with everything that is beign computed
│   ├── cross-validation - cross-validation stats 
│   ├── fda - principle components scroes and statistical analysis results
│   ├── models - saved bias detection models
│   ├── plots
```

# Pipeline
Here is a full pipeline, if you want to start from scratch. See each function's arguments to change them from default.

```
python3 ./preprocess_elan.py
python3 ./combine_open_face_output.py
python3 ./calculate_distances.py
```
Create a cross-validation config `configs/cross_validation.json` with hyperperameters you want to test.
```
python3 ./perform_cross_validation_of_bias_models.py
```
Choose the best model and add it to the `configs/models.json`
```
python3 ./remove_bias.py
python3 ./perform_fda.py
```
Now run the `statistical_analysis.R` with the appropriate model name. Look at the significance of the fpcs and update the `fpcs_plots` in `configs/fda.json` with the components that you want to plot. Now we will perform fda again.
```
python3 ./perform_fda.py
```

## Functions
### `preprocess_elan.py [--elan_fp ELAN_FP] [--meta_fp META_FP] [--videos_fp VIDEOS_FP] [--save_to SAVE_TO]`
Takes elan output and preprocess it in the following way:
* changes default column names and drops useless columns
* extracts categorical features from the video names and normalizes them
* converts timestamps from seconds to frames, based on the meta information
* shifts elan timestamps (overcoming the bug, might not be helpfull in your case)
* other custom preprocessing:
  * for each signer sets the main sign depending on what is their dominant hand
  * filters videos that have unexpected number of signs
  * maps signs with their part of speech
  * calculates the mean statustics for POS
  * finds videos and frames with/without brows movement for the bias detection model
  * extracts POS boundaries for each video

### `combine_open_face_output.py [--openface_fp OPENFACE_FP] [--elan_stats_fp ELAN_STATS_FP] [--save_to SAVE_TO]`
Combines open face files into one table, extracts categorical features from the video_names, filters the videos.

### `calculate_distances.py [--openface_fp OPENFACE_FP] [--config_fp CONFIG_FP] [--override OVERRIDE] [--save_to SAVE_TO]`
Using `configs/distances.json` calculates distances from the face points. 

Config is a dict, the keys are from:
* `perp_plane` - distance to the plane
* `perp_line` - distance to the line
* `point` - distance to the point

The values are in a list of dicts form with the following keys and values:
* `inner` - two points for the inner eyebrows 
* `outer` - two points for the outer eyebrows 
* `point` - one point to calculate distance to the point
* `perp`:
  * for distance to the line it should be two points of the line
  * for distance to the plane it should be either three points of the plane, or two points of the plane, then the fird point would be the mean, or one point of the plane, then the plane will be parallel to some axis.

See [example config](configs/distances.json).

### `perform_cross_validation_of_bias_models.py [--openface_fp OPENFACE_FP] [--elan_stats_fp ELAN_STATS_FP] [--config_fp CONFIG_FP] [--save_to SAVE_TO]`
Using `configs/cross_validation.json` performs cross validation of the bias detection models.

Config is a list of dicts, the keys of the dicts are from:
* `name` - name of the experiment, if not provided, the current time will be used
* `model_name` - a type of model from `[mlp, lasso, ridge]`
* `params` - a dict with hyperparameters of the model, the hyperparameter values should be in list form 
* `dummies` - which columns to one-hot
* `targets` - list of target distances
* `features` - list of features, if not provided, this features will be used:
```
['pose_Rx', 'pose_Rx_cos', 'pose_Tx',
 'pose_Ry', 'pose_Ry_cos', 'pose_Ty',
 'pose_Rz', 'pose_Rz_cos', 'pose_Tz']
```
* `metrics` - list of metrics, if not provided `['rmse', 'mrae', 'mae', 'mse']` will be used
* `sentence_level` - boolean, default `True`, whether to use the whole statements without eyebrow movement or all frames from statements without eyebrow movement
* `kwargs` - some other parameters to the model

See [example config](configs/cross_validation.json).

The results will be saved in `saved/cross-validation` as a separate sorted json with the experiment name and target distance, and also combined with all previous results in [logs.json](saved/cross-validation/logs.json).

### `remove_bias.py [--openface_fp OPENFACE_FP] [--elan_stats_fp ELAN_STATS_FP] [--configs_fp CONFIGS_FP] [--save_to SAVE_TO]`
Using `configs/models.json` fits and predicts the rotation bias and removes it from the distances.

Config is a list of dicts, the keys of the dicts are from:
* `name` - name of the experiment, if not provided, the current time will be used
* `model_name` - a type of model from `[mlp, lasso, ridge]`
* `params` - a dict with hyperparameters of the model
* `dummies` - which columns to one-hot
* `targets` - the target distance
* `features` - list of features, if not provided, this features will be used:
```
['pose_Rx', 'pose_Rx_cos', 'pose_Tx',
 'pose_Ry', 'pose_Ry_cos', 'pose_Ty',
 'pose_Rz', 'pose_Rz_cos', 'pose_Tz']
```
* `sentence_level` - boolean, default `True`, whether to use the whole statements without eyebrow movement or all frames from statements without eyebrow movement
* `kwargs` - some other parameters to the model

See [example config](configs/models.json).

Models will be saved in `saved/models`

### `perform_fda.py [--openface_fp OPENFACE_FP] [--pos_boundaries_fp POS_BOUNDARIES_FP] [--configs_fp CONFIGS_FP] [--save_to SAVE_TO] [--plots_save_to PLOTS_SAVE_TO]`
Using `configs/fda.json` approximates the data with functions, performs landmark registration, computes fPCs and the scores, plots the registered curves and perturbation graph of the fPCs.

Config is a list of dicts, the keys of the dicts are from:
* `experiment_name` - name of the experiment, if not provided, the current time will be used
* `n_basis` - number of basis functions to approximate points
* `order` - order of the functions to approximate points
* `n_fpca` - number of fPCs
* `names` - the target values to approximate
* `rename_dict` - rename dict for the plots
* `fpcs_plots` - optional, if provided, should be a dict with keys from `names`. The values are a list of dicts, with the keys:
  * `components` - list of indexes of components to plot
  * `plot_deaf` - boolean, default `False`, whether to plot the difference between deaf and hearing signers


See [example config](configs/fda.json).

# Plots

