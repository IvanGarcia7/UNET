import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    mse_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, true_values = batch['image'], batch['target']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_values = true_values.to(device=device, dtype=torch.float32)

            # predict the values
            predicted_values = net(image)

            print('JOJO',predicted_values[:5], true_values[:5],'JOJO')

            # compute the MSE score
            mse_score += F.mse_loss(predicted_values, true_values, reduction='mean')

    net.train()
    print('*********** ',mse_score / max(num_val_batches, 1))
    return mse_score / max(num_val_batches, 1)
