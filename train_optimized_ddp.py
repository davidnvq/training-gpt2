# torchrun --nproc_per_node=2 train_optimized_ddp.py

import torch
import torch.distributed as dist

# ! optimized step 2: enable Tensor Core usage with TF32 (only on Ampere GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # Or 'highest'

from src.utils import set_seed
from src.optimized_model import GPTModel
from src.data import create_dataloader_v1
from src.configs import GPTConfig, TrainingArguments
from src.utils import train_model_simple_with_timing


def main():
    # ! optimized step 9: use DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    set_seed(123)

    gpt_config = GPTConfig()
    training_args = TrainingArguments()

    model = GPTModel(gpt_config)
    model = torch.compile(model)  # ! optimized step 7: use torch.compile
    model.to(device).to(torch.bfloat16)  # ! optimized step 5: use bfloat16 precision

    # ! optimized step 9: use DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        fused=True,  # ! optimized step 3: use fused AdamW optimizer
    )

    train_loader, eval_loader = create_dataloader_v1(
        batch_size=training_args.batch_size,
        max_length=gpt_config.context_length,
        stride=gpt_config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        # ! optimized step 9: use DDP with DistributedSampler
        sampler=torch.utils.data.distributed.DistributedSampler,
    )

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        num_epochs=training_args.num_epochs,
        eval_freq=training_args.eval_freq,
        eval_iter=training_args.eval_iter,
    )

    # Save and load model
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        torch.save(model.state_dict(), "model.pth")
    dist.destroy_process_group()

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    main()
