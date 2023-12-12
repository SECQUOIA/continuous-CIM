python step_size.py --sde_params_dir ..parameters/paper_params --opt_params_dir ../parameters/adam --trial 0 --opt_label adam --sde MF --problem diag_scale --label paper_params
python step_size.py --sde_params_dir ../parameters/paper_params --opt_params_dir ../parameters/momentum=0.9 --trial 0 --opt_label momentum --sde MF --problem diag_scale --label paper_params
python step_size.py --sde_params_dir ../parameters/paper_params --opt_params_dir ../parameters/no_momentum --trial 0 --opt_label GD --sde MF --problem diag_scale --label paper_params
