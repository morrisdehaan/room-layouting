import os
import time
import hashlib
import argparse

def unique_save(path):
    H = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    return os.path.join(path, H)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("loc", type=str)
    args = parser.parse_args()
    print(unique_save(args.loc))