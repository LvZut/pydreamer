'''
    Make reconstructed/dream gifs using command line inputs instead of jupyter notebook.
    Runnable through ssh.
'''



import sys
import os
from os import listdir

from typing import Dict
import tempfile
from pathlib import Path
import numpy as np
from mlflow.tracking import MlflowClient

import mlflow

mlflow.get_tracking_uri()
mlflow.set_tracking_uri('file:///saivvy/pydreamer/mlruns')

FPS = 10
B, T = 5, 50

def download_artifact_npz(run_id, artifact_path) -> Dict[str, np.ndarray]:
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, artifact_path, tmpdir)
        with Path(path).open('rb') as f:
            data = np.load(f)
            return {k: data[k] for k in data.keys()}  # type: ignore

def encode_gif(frames, fps):
    # Copyright Danijar
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24', 4: 'rgba24'}[c]
    cmd = ' '.join([
        'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())  # type: ignore
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out

def make_gif(path_name, run_id, step, image_size, im_type='dream', fps=FPS):
    dest_path = f'{path_name}_{step}.gif'
    artifact = f'd2_wm_{im_type}/{step}.npz'
    data = download_artifact_npz(run_id, artifact)
    if im_type == 'dream':
        img = data['image_pred']
    else:
        img = data['image_rec']
    
    print(img.shape)
    img = img[:B, :T].reshape((-1, image_size, image_size, 3))[:,:,:,:3]
    gif = encode_gif(img, fps)
    with Path(dest_path).open('wb') as f:
        f.write(gif)
        
def make_gif_onehot(env_name, run_id, step, fps=FPS):
    dest_path = f'figures/dream_{env_name}_{step}.gif'
    artifact = f'd2_wm_dream/{step}.npz'
    data = download_artifact_npz(run_id, artifact)
    img = data['image_pred']
    print(img.shape, type(img))
    
    print(img.sum(axis=-1))
    img = img.argmax(axis=-1)
    print(img.shape, img[0,0,:20,:20])
    img = np.repeat(np.expand_dims(img * 17, axis=-1), 3, axis=-1)
    print(img.shape, img.max())
    
    
    img = img[:B, :T].reshape((-1, 256, 256, 1))[:,:,:,:3]
    gif = encode_gif(img, fps)
    with Path(dest_path).open('wb') as f:
        f.write(gif)


def process_im_num(int_num):
    # preprocess im number to fit filename
    num = str(int_num)
    if len(num) > 8 or len(num) < 4:
        exit("Invalid image number")
    
    if str(num)[-1] != 1:
        num = str(num)[:-1] + str(1)

    leading_0 = 7 - len(str(num))
    return leading_0 * "0" + num

def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result

def get_latest():
    # get id of latest run
    # stackoverflow.com/questions/2014554/find-the-newest-folder-in-a-directory-in-python

    all_subdirs = all_subdirs_of(b="mlruns/0")
    return max(all_subdirs, key=os.path.getmtime)[9:]

def get_runs():
    all_subdirs = all_subdirs_of(b="mlruns/0")
    all_subdirs = [subdir[9:] for subdir in all_subdirs]
    return all_subdirs


def main():
    # run id
    latest = get_latest()
    all_subdirs = get_runs()
    while True:
        print(f'Make image for what run?\nEnter specific run_id or leave empty (Uses: {latest})')
        run_choice = input('Run id:')
        if run_choice == '':
            run_id = latest
            break
        elif run_choice not in all_subdirs:
            print('Unable to find run id in mlruns/0/\n')
        else:
            run_id = run_choice
            break
    print(f"Using run_id: {run_id}\n")

    # image type
    while True:
        print(f'Dream (0) or Closed (1)? (default dream (0))')
        type_choice = input()
        if type_choice == '1':
            im_type = 'closed'
            break
        elif type_choice == '0' or type_choice == '':
            im_type = 'dream'
            break
        else:
            print('Invalid input\n')
    print(f"Using {im_type} image type\n")


    # image number
    im_dir = f'mlruns/0/{str(run_id)}/artifacts/d2_wm_{im_type}/'
    files = [im_dir + str(f) for f in listdir(im_dir)]
    while True:
        print(f'Image number\nCan be a single number, multiple numbers (1001, 2001, ...), "all" or leave empty (Uses last image of run)')
        im_choice = input('Image number:')
        if im_choice == "":
            latest_im = max(files, key=os.path.getmtime)

            # hard coded, get dream num from file path
            im_num = [latest_im[-11:-4]]
            print(f"Using latest: {im_num}\n")
            break
        elif im_choice == "all":
            im_num = files
            print("Using all images\n")
            break
        elif " " in im_choice:
            im_nums = im_choice.split(" ")
            im_num = []
            error = False
            for num in im_nums:
                if f'mlruns/0/{run_id}/artifacts/d2_wm_{im_type}/{process_im_num(num)}.npz' not in files:
                    error = True
                    print(f"Image {num} not found\n")
                else:
                    im_num.append(process_im_num(num))

            if not error:
                print(f"Using images: {im_num}\n")
                break
        elif f'mlruns/0/{run_id}/artifacts/d2_wm_{im_type}/{process_im_num(im_choice)}.npz' in files:
            im_num = [process_im_num(im_choice)]
            print("Using Image: {im_num}\n")
            break
        else:
            print(files)
            print('Image not found\n')

    # directory / file name
    while True:
        print("File name of image\n(e.g. dream_carla -> dream_carla_(image_number).gif)\nFiles are saved in the results/figures/ folder\nAdding '/' will create a new folder")
        im_name = input("File name:")
        if im_name.count('/') == 0:
            file_name = f'results/figures/{im_name}'
            break
        elif im_name.count('/') == 1:
            dir_name = 'results/figures/' + im_name[:im_name.find('/')+1]
            if not os.path.exists(dir_name):
                print('creating directory:', dir_name)
                os.mkdir(dir_name)
            file_name = f'results/figures/{im_name}'
            break
        elif im_name == "":
            file_name = f'results/figures/image'
            break
        else:
            print("Invalid image name\n")

    print(f"Saving image under file name: {file_name}\n")


    # image size
    image_size = input('image size (default 128):')
    
    if image_size == "":
        image_size = 128
    elif image_size.isdigit():
        image_size = int(image_size)
    else:
        print('Image size must be an Int\n')

    print(f'Using image size: {image_size}\n')


    # make gifs
    for num in im_num:
        print(num)
        num = process_im_num(num)
        print(f'Saving under name: {file_name}\nFrom file: mlruns/0/{run_id}/artifacts/d2_wm_{im_type}/{im_num}.npz\nImage_size: {image_size}')
        make_gif(file_name, run_id, num, image_size, im_type=im_type)



if __name__ == "__main__":
    main()
