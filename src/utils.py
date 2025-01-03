import torch
import random
import numpy as np
import torch.distributed as dist


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_memory_stats():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
        print(f"Allocated memory: {allocated:.4f} GB")
        print(f"Reserved memory: {reserved:.4f} GB")


def compute_loss_batch(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    return loss


def compute_loss(data_loader, model, device, num_batches=None):

    total_loss = 0.
    num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        loss = compute_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        if i >= num_batches:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, eval_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = compute_loss(train_loader, model, device, num_batches=eval_iter)
        eval_loss = compute_loss(eval_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, eval_loss


def train_model_simple_with_timing(
    model,
    train_loader,
    eval_loader,
    optimizer,
    num_epochs,
    eval_freq,
    eval_iter,
):
    device = next(model.parameters()).device

    train_losses, eval_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # Variables for cumulative average tokens/sec
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  # Ensure all prior CUDA operations are done
    t_start.record()  # Start the timer for the first interval

    # Main training loop
    for epoch in range(num_epochs):
        model.train()

        # ! optimized step 9: use DDP with DistributedSampler
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Forward and backward pass
            loss = compute_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += inp_batch.numel()

            # At evaluation intervals, measure elapsed time and tokens per second
            if global_step % eval_freq == 0:
                # End timing for the current interval
                t_end.record()
                torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                t_start.record()  # Reset timer for the next interval

                # Calculate tokens processed in this interval
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0  # Tokens per second

                # Update cumulative counters (skip the first evaluation interval)
                if global_step:  # This is False only when global_step == 0 (first evaluation)
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed

                # ! optimized step 9: use DDP
                if dist.is_initialized():
                    local_interval = tokens_interval
                    local_tensor = torch.tensor([local_interval], device=device, dtype=torch.float)
                    global_tensor = local_tensor.clone()
                    torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                    global_interval = global_tensor.item()

                    # Global tokens per second for this interval
                    global_tps = global_interval / elapsed if elapsed > 0 else 0

                    # Update cumulative tokens (local) and aggregate globally
                    cumulative_tokens += local_interval
                    local_cum_tensor = torch.tensor([cumulative_tokens], device=device, dtype=torch.float)
                    global_cum_tensor = local_cum_tensor.clone()
                    torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                    global_cumulative_tokens = global_cum_tensor.item()
                    cumulative_time += elapsed
                    global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Compute cumulative average tokens/sec (excluding the first interval)
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Evaluate model performance (this may add overhead)
                train_loss, eval_loss = evaluate_model(model, train_loader, eval_loader, device, eval_iter)
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                track_tokens.append(total_tokens)

                # ! optimized step 9: use DDP
                if dist.is_initialized():
                    avg_tps = global_avg_tps
                    tps = global_tps

                    if dist.get_rank() == 0:
                        print(f"Ep {epoch+1}, Step {global_step:06d}, "
                              f"Train: {train_loss:.3f}, Val: {eval_loss:.3f}, "
                              f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}")
                else:

                    print(f"Ep {epoch+1}, Step {global_step:06d}, "
                          f"Train: {train_loss:.3f}, Val: {eval_loss:.3f}, "
                          f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}")

        if dist.is_initialized() and dist.get_rank() == 0:
            print_memory_stats()

    return train_losses, eval_losses, track_tokens
