import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop)
    val_loader = get_loader(config.val_dir, config.batch_size, config.len_crop)

    solver = Solver(vcc_loader, val_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./spmel')
    parser.add_argument('--val_dir', type=str, default='./spmel_val')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of iterations per epoch')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--val_iters', type=int, default=20, help='number of total epochs')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--checkpoint', type=str, default=None, help='load from checkpoint')
    parser.add_argument('--save_path', type=str, default='trained_model.ckpt', help='checkpoint save path')
    parser.add_argument('--log_path', type=str, default='./log', help='log save path')

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)