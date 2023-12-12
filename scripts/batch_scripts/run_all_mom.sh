declare -a problems=("diag_scale" "BoxQP") 
declare -a settings=("paper_params")

for problem in "${problems[@]}";do
    for setting in "${settings[@]}";do
        python single_parameter_setting.py --sde_params_dir ../parameters/$setting --opt_params_dir ../parameters/momentum=0.9 --opt_label momentum --sde MF --problem $problem --label $setting --stride 100
        python plot_single_parameter_setting.py --opt_label momentum --sde MF --problem $problem --label $setting

    done
done