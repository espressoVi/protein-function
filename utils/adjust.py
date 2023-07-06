#!/usr/bin/env python
import numpy as np
import sys
import os

def modify(number, adjustment):
    if adjustment > 0:
        return min(1.0, number+adjustment)
    if adjustment < 0:
        return max(0.0001, number+adjustment)
def adjust(path, adjustment):
    with open(path, 'r') as f:
        raw = f.readlines()
    row = [i.strip().split('\t') for i in raw]
    row = [f"{i[0]}\t{i[1]}\t{modify(float(i[2]),adjustment):.4f}" for i in row]
    final = '\n'.join(row)
    with open(path, 'w') as f:
        f.writelines(final)

def main():
    path = sys.argv[1]
    adjustment = float(sys.argv[2])
    assert os.path.exists(path)
    print(path, adjustment)
    adjust(path, adjustment)
    
if __name__ == "__main__":
    main()
