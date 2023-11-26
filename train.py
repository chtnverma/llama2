import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
from timeit import default_timer as timer
from model import ImgToTextHfLlama2Decoder
from data import FlickrDataset, tokenize_and_collate
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import trackers

# To run:
# python -u train.py --log-interval 5 --max-train-steps 5 --max-eval-steps 2


def timeit():
    torch.cuda.synchronize()
    return timer()


def train(args, model, device, train_loader, optimizer, epoch, wandb_tracker):
    model.train()
    time_dict = defaultdict(float)
    starttime_all = timeit()
    num_steps = 0
    losses = []
    print('\nTraining')
    for batch_idx, (image_emb, token_ids, target_ids) in enumerate(train_loader):
        print(f'Batch num = {batch_idx}', end="\r", flush=True)
        if num_steps >= args.max_train_steps:
            break
        image_emb, token_ids, target_ids = image_emb.to(device), token_ids.to(device), target_ids.to(device)
        optimizer.zero_grad()
        
        # Forward.
        starttime_forward = timeit()
        loss, _ = model(image_emb, token_ids, target_ids)
        endtime_forward = timeit()
        time_dict["forward"] += endtime_forward - starttime_forward
        
        # Backward.
        loss.backward()
        losses.append(loss.detach().item())
        optimizer.step()
        endtime_backward = timeit()
        time_dict["backward"] += endtime_backward - endtime_forward

        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(
                __name__,
                epoch, batch_idx * len(image_emb), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum(losses) / len(losses) ))

            wandb_tracker.log({'train_loss': sum(losses) / len(losses)}, step=(batch_idx+1) * args.batch_size + len(train_loader.dataset) * epoch)

            time_dict["all"] = timeit() - starttime_all
            starttime_all = timeit()
            print(f"[TRAIN] ******** Overall Speeds ********")
            for k, v in time_dict.items():
                print(f"[TRAIN] {k}: {v / args.log_interval} seconds / batch")
            print("\n")
            time_dict = defaultdict(float)
        num_steps += 1


def test(args, model, device, test_loader, wandb_tracker):
    model.eval()
    test_loss = 0
    print('\nTesting')
    losses = []
    num_steps = 0
    with torch.no_grad():
        for image_emb, token_ids, target_ids in test_loader:
            if num_steps >= args.max_eval_steps:
                break
            image_emb, token_ids, target_ids = image_emb.to(device), token_ids.to(device), target_ids.to(device)
            with torch.no_grad():
                loss, _ = model(image_emb, token_ids, target_ids)
            losses.append(loss)
            num_steps += 1

    test_loss = sum(losses) / len(losses)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    wandb_tracker.log({'test_loss': test_loss})


def main():
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
    parser.add_argument('--generate-tokens', action='store_true', default=False,
                        help='Generate tokens for a random image.')
    parser.add_argument('--model-path-to-load', default="",
                        help='For generation: provide local path')
    parser.add_argument('--temperature', default=0.0,
                        help='For generation: provide temperature')
    args = parser.parse_args()
    
    
    # Misc setup.
    torch.manual_seed(args.seed)
    wandb_tracker = trackers.Tracker(vars(args))
    device = torch.device("cuda")
    print('Using device:', device)
    
    # Data.
    dataset = FlickrDataset(
        annotations_file='../data/results.csv', 
        img_dir='../data/flickr30k_images')
    train_ds, test_ds = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=tokenize_and_collate)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=tokenize_and_collate)
    
    # Generate (inference only).
    if args.generate_tokens:
        assert args.model_path_to_load != "", "Model path must be set for generation"
        random_index = int(torch.rand((1)).item()*len(train_ds))
        image_emb, image_path, actual_text = train_ds[random_index]
        print(f"Image: {image_path}")
        print(f"Actual text: {actual_text}")
        print("Generated text =")
        ImgToTextHfLlama2Decoder.generate(image_emb, model_path=args.model_path_to_load, temperature=args.temperature)
        return

    # Training loop.
    model = ImgToTextHfLlama2Decoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, 10):
        print(f"Starting epoch # {epoch}")
        train(args, model, device, train_dataloader, optimizer, epoch, wandb_tracker)
        test(args, model, device, test_dataloader, wandb_tracker)
        scheduler.step()
        # torch.save(model, f'models/model_{epoch}.pt')

    torch.save(model, f'models/model_{epoch}.pt')
    

if __name__ == "__main__":
    main()