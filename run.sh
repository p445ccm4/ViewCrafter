python inference.py \
--image_dir inputs/b3.png \
--out_dir ./outputs/b3 \
--traj_txt test/trajs/pan.txt \
--mode 'single_view_txt' \
--center_scale 1. \
--elevation=5 \
--ckpt_path ./checkpoints/model.ckpt \
--config configs/inference_pvd_1024.yaml \
--ddim_steps 50 \
--video_length 25 \
--device 'cuda:0' \
--height 576 --width 1024 \
--model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
# --seed 123 \
# --d_theta -30  \
# --d_phi 45 \
# --d_r -.2   \
# --d_x 50   \
# --d_y 25   \