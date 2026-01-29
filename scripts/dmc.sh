# Walker Walk
for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    # PoLiCER
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_PoLiCER.py reward_stack=true time_shift=2 time_crop=2 'task@_global_=walker_walk' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 reward_max_episodes=30 num_interact=30000 max_feedback=200 reward_batch=10 reward_update=20 increase_q=5 init_k=25 aug_ratio=10
    # PEBBLE
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_PEBBLE.py reward_stack=true 'task@_global_=walker_walk' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 num_interact=30000 max_feedback=200 reward_batch=10 reward_update=20
    # QPA
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_QPA.py reward_stack=true time_shift=2 time_crop=2 'task@_global_=walker_walk' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 num_interact=30000 max_feedback=200 reward_batch=10 reward_update=20
    # SURF
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_SURF.py reward_stack=true time_shift=2 time_crop=2 'task@_global_=walker_walk' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 num_interact=30000 max_feedback=200 reward_batch=10 reward_update=20 inv_label_ratio=5 threshold_u=0.99 lambda_u=0.1
    # DrQ-v2
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_drqv2.py 'task@_global_=walker_walk' seed=$seed num_train_frames=1000000
done

#Cheetah Run
for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    # PoLiCER
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_PoLiCER.py reward_stack=true time_shift=5 time_crop=5 'task@_global_=cheetah_run' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 reward_max_episodes=30 num_interact=30000 max_feedback=1000 reward_batch=50 reward_update=20 increase_q=5 init_k=25 aug_ratio=10
    # PEBBLE
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_PEBBLE.py reward_stack=true 'task@_global_=cheetah_run' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 num_interact=30000 max_feedback=1000 reward_batch=50 reward_update=20
    # QPA
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_QPA.py reward_stack=true time_shift=5 time_crop=5 'task@_global_=cheetah_run' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 num_interact=30000 max_feedback=1000 reward_batch=50 reward_update=20
    # SURF
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_SURF.py reward_stack=true time_shift=5 time_crop=5 'task@_global_=cheetah_run' seed=$seed num_train_frames=1000000 \
    num_unsup_frames=0 num_interact=30000 max_feedback=1000 reward_batch=50 reward_update=20 inv_label_ratio=5 threshold_u=0.99
    # DrQ-v2
    CUDA_VISIBLE_DEVICES=$device EGL_DEVICE_ID=$device python train_drqv2.py 'task@_global_=cheetah_run' seed=$seed num_train_frames=1000000
done