import os
import time
import hashlib
import argparse

def unique_save(path):
    H = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:5]
    if args.loc[-1] in ("/", "\\"):
        return os.path.join(path, H)
    else:
        return f"{path}_{H}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("loc", type=str)
    args = parser.parse_args()
    print(unique_save(args.loc))