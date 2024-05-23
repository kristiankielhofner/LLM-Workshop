#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2
##SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --partition=batch
#SBATCH --gres=gpu:2 # Adjust number of GPUs here
#SBATCH --output=hello_world.out
#SBATCH --err=hello_world.err

# Bail on error
set -e

echo "START TIME: $(date)"
echo "Hostname is $HOSTNAME"

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
#source ~/.bashrc
#source /etc/profile.d/conda.sh
#conda init
#source activate base
#conda activate /local/mgpu/conda

echo "Using Python: $(which python3)"

# netrc for wandb
if [ -r /local/mgpu/netrc ]; then
    cp /local/mgpu/netrc /root/.netrc
else
    WANDB_DISABLED=true
fi

# Attempt to add current directory as safe to git to silence warnings
git config --global --add safe.directory ${PWD} || true

WORK_DIR=${PWD}/chat_assistant/sft/training
cd ${WORK_DIR}

# have the below in case of debugging nccl issues such as nccl timeout.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
#export NCCL_PROTO=simple
#export RDMAV_FORK_SAFE=1
#export FI_EFA_FORK_SAFE=1
#export FI_EFA_USE_DEVICE_RDMA=1
#export FI_PROVIDER=efa
#export FI_LOG_LEVEL=1
#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=ens

GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(($NNODES * $GPUS_PER_NODE))

# Get master node address for accelerate
MASTER_ADDR=$(hostname -i)

# Generate random master port from ephemeral range
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Master socket is ${MASTER_ADDR}:${MASTER_PORT}"

# OTHER LAUNCHERS CAN BE USED HERE
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
LAUNCHER="accelerate launch \
    --config_file configs/fsdp_config_qlora.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

PROGRAM="\
    train.py \
        --seed 100 \
        --model_name_or_path "mistralai/Mistral-7B-v0.1" \
        --dataset_name "smangrul/ultrachat-10k-chatml" \
        --chat_template_format "chatml" \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "train,test" \
        --max_seq_len 2048 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --log_level "info" \
        --logging_strategy "steps" \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --push_to_hub \
        --hub_private_repo True \
        --hub_strategy "every_save" \
        --bf16 True \
        --packing True \
        --learning_rate 1e-4 \
        --lr_scheduler_type "cosine" \
        --weight_decay 1e-4 \
        --warmup_ratio 0.0 \
        --max_grad_norm 1.0 \
        --output_dir "llama-sft-qlora-fsdp" \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing True \
        --use_reentrant True \
        --dataset_text_field "content" \
        --use_flash_attn True \
        --use_peft_lora True \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_target_modules "all-linear" \
        --use_4bit_quantization True \
        --use_nested_quant True \
        --bnb_4bit_compute_dtype "bfloat16" \
        --bnb_4bit_quant_storage_dtype "bfloat16"
"

# Put final command together
CMD="$LAUNCHER $PROGRAM"

# Export everything to be sure
set -a

echo "Current environment:"
set

echo "Launching command: $CMD"
#srun --nodes ${NNODES} --ntasks-per-node ${SLURM_NTASKS_PER_NODE} --jobid ${SLURM_JOBID} bash -c "$CMD"
srun --jobid $SLURM_JOBID bash -c "$CMD"
echo "END TIME: $(date)"