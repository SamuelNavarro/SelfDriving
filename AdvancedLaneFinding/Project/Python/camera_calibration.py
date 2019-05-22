import numpy as np
import cv2
import glob
import matploblib.pyplot as plt
from Pathlib import Path


import argparse


def calibrate_camera(self, imgs_path, nx, ny):
    images = glob.glob(imgs_path)
    objpoints, imgpoints = [], []
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints



def get_undist_img(







def main(args):
    input_path = Path(args.input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='The path where the images or video are located')

# def main(args):

    # input_path = Path(args.input)
    # output = Path(args.output)
    # assert input_path.exists(), f'Error: {input_path} does not exist.'
    # output.mkdir(exist_ok=True)

    # train_path = output.joinpath('train.csv')
    # val_path = output.joinpath('val.csv')
    # text_iter = get_texts(input_path)
    # write_file(train_path, text_iter, args.num_tokens)
    # write_file(val_path, text_iter, args.num_tokens / 10)


# if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', required=True,
                        # help='the directory where the Wikipedia data extracted '
                             # 'with WikiExtractor.py is located. Consists of '
                             # 'directories AA, AB, AC, etc.')
    # parser.add_argument('-o', '--output', required=True,
                        # help='the output directory where the merged Wikipedia '
                             # 'documents should be saved')
    # parser.add_argument('-n', '--num-tokens', type=int, default=100000000,
                        # help='the # of tokens that the merged document should '
                             # 'contain (default: 100M)')
    # args = parser.parse_args()
    # main(args)
