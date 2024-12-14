import torch


def load(model: torch.nn.Module, checkpoint_path: str, device: torch.device, strict: bool = False):
    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    optimizer = checkpoint.get('optimizer', None)

    model_dict = model.state_dict()
    compatible_weights = {k: v for k, v in state_dict.items()
                          if k in model_dict}
    if len(compatible_weights) != 0:
        model.load_state_dict(compatible_weights, strict=strict)
        print(f'=> Loaded {len(compatible_weights)}/{len(model_dict)} layers')
    else:
        print('=> No checkpoint. Initializing model from scratch')

    return epoch, loss, optimizer


def save(model, optimizer: torch.optim.Optimizer, epoch: int, epoch_loss: dict, checkpoint_path: str):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': epoch_loss,
    }, checkpoint_path)
