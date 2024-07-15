import os

if __name__ == "__main__":
    os.system('python src\pretrain.py')
    os.system('python src\tune.py')
    os.system('python src\evaluate.py')
