from train import main
import argparse

# take all arguments from cli and create a dictionary with them

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--pos_weight', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--bn_momentum', type=float, required=True)
parser.add_argument('--loss_type', type=str, required=True)

config = parser.parse_args()
main(config)