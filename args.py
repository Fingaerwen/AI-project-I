import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument('--backbone', type=str, default='fasterrcnn_resnet50_fpn', choices=['fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3'])
    
    parser.add_argument('--csv-dir', type=str, default='./data/CSVs')
    parser.add_argument('--out-dir', type=str, default='./sessions')

    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.1e-4)
    
    return parser.parse_args()
