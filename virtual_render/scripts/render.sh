version=1024
seed=123
scene_name=$1

res_dir="./results/"

echo "scene_name: $1"

ckpt=./checkpoints/1024_mdm/1024-mdm-checkpoint.ckpt

config=configs/stage2-1024_mdm_waymo_infer.yaml # model with class-embedding for color
val_files=./virtual_render/virtual_data/${scene_name}-virtual_data_frames.json
echo "val_files: ${val_files}"

#########################################################################
name=render_data/${scene_name}_color_virtual_pose_render_left_2m

python3 virtual_render/virtual_pose_render.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 576 --width 1024 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--text_input \
--video_length 16 \
--fps 10 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae \
--val_files $val_files
