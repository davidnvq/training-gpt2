import torch

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
    set_seed(123)

    gpt_config = GPTConfig()
    training_args = TrainingArguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(gpt_config)
    model.to(device).to(torch.bfloat16)  # ! optimized step 5: use bfloat16 precision

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
    torch.save(model.state_dict(), "model.pth")

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    main()
