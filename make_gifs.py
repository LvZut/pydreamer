'''
    Make dream gifs using command line arguments instead of jupyter notebook.
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

# mlflow.get_tracking_uri()
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

def make_gif(path_name, run_id, step, image_size, fps=FPS):
    dest_path = f'{path_name}_{step}.gif'
    artifact = f'd2_wm_dream/{step}.npz'
    data = download_artifact_npz(run_id, artifact)
    img = data['image_pred']
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


def process_dream_num(int_num):
    # preprocess dream number to fit filename
    num = str(int_num)
    if len(num) > 7 or len(num) < 1:
        exit("Invalid dream number")
    
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




# args: dream_name, dream number, run id
def main(args):
    # get run id
<<<<<<< HEAD
    if len(args) > 2 and str(args[2]) == 'latest':
        run_id = get_latest()
    elif len(args) > 2:
        if len(args) == 32:
=======

    if len(args) > 2:
        if str(args[2]) == 'latest':
            run_id = get_latest()
        if len(args[2]) == 32:
>>>>>>> f926bd1373466388397b3bf10d4456083afa2d1c
            run_id = str(args[2])
        else:
            exit("invalid run id")
    else:
        # get last modified run if no run id is given
        run_id = get_latest()

    
    dirname = os.path.dirname(__file__)
    if len(args) > 0:
        file_name = f"results/atari/figures/{args[0]}"

        # create directory if '/' in dreamname is given
        file_name = os.path.join(dirname, file_name)

        if args[0].find('/') != -1:
            dir_name = file_name[:-(args[0].find('/')+2)]
            print(dir_name, file_name, '\n\n')
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
    else:
        file_name = "results/atari/figures/dream"

    if len(args) > 3:
        if not args[3].isnumeric():
            exit("invalid image size")
        image_size = int(args[3])
    else:
        image_size = 64

    if len(args) > 1:
        dream_num = process_dream_num(args[1])
    else:
        dream_dir = f'mlruns/0/{str(run_id)}/artifacts/d2_wm_dream'
        files = [f for f in listdir(dream_dir)]

        for dream_file in files:
            dream_num = dream_file[:7]
            make_gif(file_name, run_id, dream_num, image_size)

        exit('done')
    


    make_gif(file_name, run_id, dream_num, image_size)





if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f'usage: make_gif.py dream_name dream_number run_id image_size\n\
                default:\n\
                dream_name: results/atari/figures/dream_"step" (adding / to dream_name creates dir)\n\
                dream_number: all dreams found in corresponding dreams dir\n\
                run_id: last edited directory inside of mlruns/0, same as "latest"\n\
                image_size: 64')
        exit(0)
    main(sys.argv[1:])
