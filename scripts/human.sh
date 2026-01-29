# Walker Walk
for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    # PoLiCER
    python train_PoLiCER_human.py env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=500000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=100 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=5.0 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0

    # PEBBLE
    python train_PEBBLE_human.py env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 num_train_steps=500000 agent.params.batch_size=1024 double_q_critic.params.hidden_dim=1024 double_q_critic.params.hidden_depth=2 diag_gaussian_actor.params.hidden_dim=1024 diag_gaussian_actor.params.hidden_depth=2 \
    num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=100 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0
done

# Window Open
unset LD_PRELOAD
export LD_PRELOAD=""

for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    # PoLiCER
    python train_PoLiCER_human.py env=metaworld_window-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_train_steps=500000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=512 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=512 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=15 num_interact=10000 max_feedback=240 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    use_pls_sampling=true use_crop_aug=true tau_min=1.0 tau_max=1.0 tau_delta=0.0 qreset=true rreset=3 init_k=25 increase_q=7.5 step_k=0.9 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0 num_eval_episodes=10

    # PEBBLE
    python train_PEBBLE_human.py env=metaworld_window-open-v2 seed=$seed agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 num_train_steps=500000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=512 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=512 diag_gaussian_actor.params.hidden_depth=3 \
    num_unsup_steps=9000 reward_batch=10 num_interact=20000 max_feedback=100 feed_type=1 reward_update=200 segment=50 max_reward_buffer_size=100 \
    teacher_beta=-1 teacher_gamma=1 teacher_eps_skip=0 teacher_eps_mistake=0 teacher_eps_equal=0 num_eval_episodes=10
done
