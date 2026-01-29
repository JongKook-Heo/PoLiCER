# Window Open
for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    # PoLiCER
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device  python train_PoLiCER.py reward_stack=true time_shift=2 time_crop=2 'task@_global_=metaworld_window-open' seed=$seed num_train_frames=800000 \
    num_unsup_frames=0 reward_max_episodes=30 num_interact=10000 max_feedback=800 reward_batch=20 reward_update=40 increase_q=7.5 init_k=25 aug_ratio=10 max_rr=2 \
    segment=25 
    # PEBBLE
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device  python train_PEBBLE.py reward_stack=true 'task@_global_=metaworld_window-open' seed=$seed num_train_frames=800000 \
    num_unsup_frames=0 num_interact=10000 max_feedback=800 reward_batch=20 reward_update=40 \
    segment=25
    # SURF
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device  python train_SURF.py reward_stack=true time_shift=2 time_crop=2 'task@_global_=metaworld_window-open' seed=$seed num_train_frames=800000 \
    num_unsup_frames=0 num_interact=10000 max_feedback=800 reward_batch=20 reward_update=40 threshold_u=0.99 lambda_u=0.1 \
    segment=25
    # QPA
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device  python train_QPA.py reward_stack=true time_shift=2 time_crop=2 'task@_global_=metaworld_window-open' seed=$seed num_train_frames=800000 \
    num_unsup_frames=0 num_interact=10000 max_feedback=800 reward_batch=20 reward_update=40 \
    segment=25
    # DrQ-v2
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device  python train_drqv2.py 'task@_global_=metaworld_window-open' seed=$seed num_train_frames=800000
done


