uv run eval_aime24.py --model /dev/shm/rwkv7-g1k-13.3b-20260930-ctx25600.pth --bsz 960 --rollout 32 --ks 1,2,4,8,16,32 --prompt-style base --out-dir ./aime24_runs_comp/g1k_13p3b_base
uv run eval_aime24.py --model /dev/shm/rwkv7-g1k-13.3b-20260930-ctx25600.pth --bsz 960 --rollout 32 --ks 1,2,4,8,16,32 --prompt-style think --out-dir ./aime24_runs_comp/g1k_13p3b_think
uv run eval_aime24.py --model /dev/shm/rwkv7-g1k-13.3b-20260930-ctx25600.pth --bsz 960 --rollout 32 --ks 1,2,4,8,16,32 --prompt-style fake_think --out-dir ./aime24_runs_comp/g1k_13p3b_fakethink
