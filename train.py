import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import torch.nn.functional as F
from timeit import default_timer as timer
from model import FakeModel, ImgToTextHfLlama2Decoder
from data import FlickrDataset, transform_raw_image, collate_with_tokenize, FakeDataset
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import torch.autograd.profiler as profiler
import trackers

# To run:
# python -u train.py --log-interval 5 --max-train-steps 5 --max-eval-steps 2

# Training settings
parser = argparse.ArgumentParser(description='img2txt')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--log-interval', type=int, default=1,
                    help='number of batches to log after (default: 1)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--max-train-steps', type=int, default=10, metavar='ST',
                    help='max train steps per epoch (default: 10)')
parser.add_argument('--max-eval-steps', type=int, default=10, metavar='ST',
                    help='max train steps per epoch (default: 10)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()
torch.manual_seed(args.seed)
wandb_tracker = trackers.Tracker(vars(args))


dataset = FlickrDataset(
    annotations_file='../data/results.csv', 
    img_dir='../data/flickr30k_images', 
    transform=transform_raw_image)
# dataset = FakeDataset()
train_ds, test_ds = random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_with_tokenize)
test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_with_tokenize)

def timeit(device):
    if True: #device == torch.device("mps"):
        torch.cuda.synchronize()
    return timer()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    time_dict = defaultdict(float)
    st_all = timeit(device)
    # st_r = st_all
    num_steps = 0
    losses = []
    for batch_idx, (image_emb, token_ids) in enumerate(train_loader):
        # print(f'Batch num = {batch_idx}')
        if num_steps >= args.max_train_steps:
            break
        # time_dict["next_example"] += timeit(device) - st_r
        # print(f"image_emb = {image_emb}")
        # print(f"token_ids = {token_ids}")
        image_emb, token_ids = image_emb.to(device), token_ids.to(device)
        optimizer.zero_grad()
        # st_f = timeit(device)
        logits, loss = model(image_emb, token_ids)
        # print("Forward complete.")
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        # et_f = timeit(device)
        # print(f"[{__file__.split('/')[-1]}] At overall level: time for forward() = {et_f - st_f}")
        # time_dict["forward"] += et_f - st_f
        # st_b = timeit(device)
        
        if isinstance(model, torch.nn.DataParallel):
            loss = model.module.train_val_step(logits, loss, token_ids, train=True)
        else:
            loss = model.train_val_step(logits, loss, token_ids, train=True)
        losses.append(loss.item())
        # print("Train step complete.")
            
            
        # et_b = timeit(device)
        # print(f"[{__file__.split('/')[-1]}] At overall level: time for backward() = {et_b - st_b}")
        # time_dict["backward"] += et_b - st_b

        # st_o = timeit(device)
        # accelerator.backward(loss)
        optimizer.step()
        # time_dict["optimizer"] += timeit(device) - st_o

        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(
                __name__,
                epoch, batch_idx * len(image_emb), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum(losses) / len(losses) ))

            wandb_tracker.log({'train_loss': sum(losses) / len(losses)}, step=(batch_idx+1) + len(train_loader.dataset) * epoch)

            time_dict["all"] = timeit(device) - st_all
            st_all = timeit(device)
            print(f"[TRAIN] ******** Overall Speeds ********")
            for k, v in time_dict.items():
                if k == "all":
                    print(f"{k}: {v / args.log_interval} seconds / batch")
                    continue
                print(f"[TRAIN] {k}: {v / args.log_interval} seconds / batch")
            print("\n")
            time_dict = defaultdict(float)
        
        num_steps += 1
        st_r = timeit(device)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    print('Testing')
    losses = []
    num_steps = 0
    with torch.no_grad():
        for image_emb, token_ids in test_loader:
            if num_steps >= args.max_eval_steps:
                break
            image_emb, token_ids = image_emb.to(device), token_ids.to(device)
            logits, loss = model(image_emb, token_ids)
            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.train_val_step(logits, loss, token_ids, train=False)
            else:
                loss = model.train_val_step(logits, loss, token_ids, train=False)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            losses.append(loss)
            num_steps += 1

    test_loss = sum(losses) / len(losses)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    wandb_tracker.log({'test_loss': test_loss})

if use_cuda:
    device = torch.device("cuda")
    print("## Using gpu")
elif use_mps:
    device = torch.device("mps")
    print("## Using mps")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")


# accelerator = Accelerator()
# device = accelerator.device

print('Using device:', device)
model = ImgToTextHfLlama2Decoder().to(device)
# model= torch.nn.DataParallel(model)

# model = FakeModel(2048, 768).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-5)
scheduler = StepLR(optimizer, step_size=1)

# model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

for epoch in range(1, 10):
    print(f"Starting epoch # {epoch}")
    train(args, model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)
    scheduler.step()
    torch.save(model, f'models/model_{epoch}.pt')