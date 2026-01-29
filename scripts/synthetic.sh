for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
# Walker Walk
    python train_PoLiCER.py env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=1000000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=100 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=5.0 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

# Cheetah Run
    python train_PoLiCER.py env=cheetah_run seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=1000000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=100 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=5.0 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

# Humanoid Stand
    python train_PoLiCER.py env=humanoid_stand seed=$seed agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 num_train_steps=2000000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=50 num_interact=5000 max_feedback=10000 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=1.43 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

# Sweep Into
    python train_PoLiCER.py env=metaworld_sweep-into-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003  num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=50 num_interact=5000 max_feedback=5000 feed_type=1 reward_update=200 segment=25 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=7.5 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

# Door Open
    python train_PoLiCER.py env=metaworld_door-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003  num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=40 num_interact=5000 max_feedback=4000 feed_type=1 reward_update=200 segment=25 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=7.5 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

# Drawer Open
    python train_PoLiCER.py env=metaworld_drawer-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003  num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=50 num_interact=5000 max_feedback=5000 feed_type=1 reward_update=200 segment=25 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=7.5 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

# Hammer
    python train_PoLiCER.py env=metaworld_hammer-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003  num_train_steps=2000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=50 num_interact=5000 max_feedback=10000 feed_type=1 reward_update=200 segment=25 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=7.5 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0
done