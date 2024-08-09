import os
import argparse

def copy_sub_dirs(partial_path, src_dir, target_dir):
    full_path = os.path.join(src_dir, partial_path)
    for d in os.listdir(full_path):
        if os.path.isdir(os.path.join(full_path, d)):
            to_path = os.path.join(target_dir, partial_path, d)
            os.mkdir(to_path)
            copy_sub_dirs(os.path.join(partial_path, d), src_dir, target_dir)

def main():
    parser = argparse.ArgumentParser(description='Copy directories')
    parser.add_argument('--src_dir', type=str, help='source directory')
    parser.add_argument('--target_dir', type=str, help='target directory')
    args = parser.parse_args()
    print("Will copy dirs")
    copy_sub_dirs("", args.src_dir, args.target_dir)
    print("Copying done")