# README

The following repository consists of a concise version of the backend code for the publication:

**Classification of substances by health hazard using deep neural networks and molecular electron densities.**
by Singh et al., 2024

The data required for the ECHA dataset is present in the following files:

* Training_data_ECHA_unsorted.csv
* Test_data_ECHA_unsorted.csv
* Val_data_ECHA_unsorted.csv

The directory format required for the cube files is shown in the directory **ECHA_cubes**
and **tox_datasets**.

The dataloaders for the two datasets are written in *open_foodtox_dataloader.py* and *custom_loader.py*.

As shown in the code, the files are expected to have a certain nomenclature.
For eDen files, this is expected to be: ***index_e_Dens_vol_data_.npy*** and for ENeg files, it's expected to be ***index_e_Eneg_vol_data_.npy*** where *index* refers to the integer index of the files, starting at 0. 

Moreover, all the voxel shapes for the files are expected in a ***mode_vox_info.npy*** file where *mode* refers to either: *train*, *test* or *val*.

The training (and optimisation) loops are present in *open_foodtox_optimise.py* and *Optuna_tuned_Echa.py*.

The SMILES strings for both datasets are present in files: *CompFood.csv* and ECHA csv files split into each of the train/test/val sets with allowed or prohibited labels.
