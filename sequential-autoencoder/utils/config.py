import argparse


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch Recipe1M Training')

    # Model options
    parser.add_argument('--seed', default=42, type=int)

    # Model training options
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--evaluate_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    
    # TODO: Reverify this later, once things are loaded
    parser.add_argument('--vocab_size', default=9226, type=int)


    # Model architecture options
    parser.add_argument('--image_encoder', default='resnet', type=str)
    parser.add_argument('--hidden_dim', default=512, type=float)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--batch_loss_weight', default=1.0, type=float)
    parser.add_argument('--ce_loss_weight', default=1.0, type=float)
    
    # Model saving and checkpointing options
    parser.add_argument('--save_dir', default='checkpoints', type=str)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--resume_path', default=None, type=str)

    # Model evaluation options
    parser.add_argument('--retrieval_mode', default='image2recipe', type=str, choices=['image2recipe', 'recipe2image'])
    parser.add_argument('--medr_N', default=1000, type=int)

    args = parser.parse_args()

    return args