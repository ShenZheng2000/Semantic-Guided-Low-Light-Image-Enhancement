import argparse

parser = argparse.ArgumentParser(description="Train_Test")

### Training Settings
# General Parameters
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=6)
parser.add_argument('--val_batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshot_iter', type=int, default=10)
parser.add_argument('--scale_factor', type=int, default=1)
parser.add_argument("--num_of_SegClass", type=int, default=21, help='Number of Segmentation Classes, default VOC = 21')

# Weight Parameters
parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
parser.add_argument('--snapshots_folder', type=str, default="weight/")
parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--pretrain_dir', type=str, default="weight/Epoch99.pth")

# Ablation Parameters
parser.add_argument('--conv_type', type=str, default="dsc", choices=['dsc', 'dc', 'tc'])
parser.add_argument('--patch_size', type=int, default=4, choices=[3, 4, 5])
parser.add_argument('--exp_level', type=float, default=0.6, choices=[0.5, 0.6, 0.7])

### Testing Settings
parser.add_argument('--weight_dir', type=str, help='directory for pretrained weight', default='weight/Epoch99.pth')
parser.add_argument('--test_dir', type=str, help='directory for testing output', default='test_output')
parser.add_argument('--input_dir', type=str, help='directory for testing input', default='data/test_data/')

args = parser.parse_args()
