from typing import Optional
import argparse
import pathlib
import subprocess
from collections import defaultdict
from typing import Iterator
import numpy as np
import time
import torch
from torch import tensor, Tensor
import torch.nn as nn
import torch.distributions as D
from torch.profiler import ProfilerActivity
import mlflow

import tools
from tools import mlflow_start_or_resume, param_count, NoProfiler
from data import OfflineDataSequential, OfflineDataRandom
from preprocessing import MinigridPreprocess
from models import *
from modules_io import *
from modules_mem import *
from modules_tools import *

torch.distributions.Distribution.set_default_validate_args(False)
torch.backends.cudnn.benchmark = True  # type: ignore


def run(conf):
    assert not(conf.keep_state and not conf.data_seq), "Should train sequentially if keeping state"

    if conf.generator_run:
        run_generator(conf)

    mlflow_start_or_resume(conf.run_name, conf.resume_id)
    mlflow.log_params(vars(conf))
    device = torch.device(conf.device)

    data = (OfflineDataSequential(conf.input_dir) if conf.data_seq else OfflineDataRandom(conf.input_dir))
    data_eval = (OfflineDataSequential(conf.eval_dir) if conf.data_seq else OfflineDataRandom(conf.eval_dir))

    preprocess = MinigridPreprocess(image_categorical=conf.image_channels if conf.image_categorical else None,
                                    image_key=conf.image_key,
                                    map_categorical=conf.map_channels,
                                    map_key=conf.map_key,
                                    device=device)

    state_dim = conf.deter_dim + conf.stoch_dim + conf.global_dim

    # Encoder

    if conf.image_encoder == 'cnn':
        encoder = ConvEncoder(in_channels=conf.image_channels,
                              out_dim=conf.embed_dim)
    else:
        encoder = DenseEncoder(in_dim=conf.image_size * conf.image_size * conf.image_channels,
                               out_dim=conf.embed_dim,
                               hidden_layers=conf.image_encoder_layers)

    # Decoder

    if conf.image_decoder == 'cnn':
        decoder = ConvDecoder(in_dim=state_dim,
                                 out_channels=conf.image_channels)
    else:
        decoder = DenseDecoder(in_dim=state_dim,
                               out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                               hidden_layers=conf.image_decoder_layers,
                               min_prob=conf.image_decoder_min_prob)

    # Map decoder

    if conf.map_model == 'vae':
        map_model = CondVAEHead(
            encoder=DenseEncoder(in_dim=conf.map_size * conf.map_size * conf.map_channels,
                                 out_dim=conf.embed_dim,
                                 hidden_layers=3),
            decoder=DenseDecoder(in_dim=state_dim + conf.map_stoch_dim,
                                 out_shape=(conf.map_channels, conf.map_size, conf.map_size),
                                 hidden_layers=4),
            state_dim=state_dim,
            latent_dim=conf.map_stoch_dim
        )
    elif conf.map_model == 'direct':
        map_model = DirectHead(
            decoder=DenseDecoder(in_dim=state_dim,
                                 out_shape=(conf.map_channels, conf.map_size, conf.map_size),
                                 hidden_dim=conf.map_hidden_dim,
                                 hidden_layers=conf.map_hidden_layers),
        )
    else:
        map_model = NoHead(out_shape=(conf.map_channels, conf.map_size, conf.map_size))

    # Memory model

    if conf.mem_model == 'global_state':
        mem_model = GlobalStateMem(embed_dim=conf.embed_dim,
                                   action_dim=conf.action_dim,
                                   mem_dim=conf.deter_dim,
                                   stoch_dim=conf.global_dim,
                                   hidden_dim=conf.hidden_dim,
                                   loss_type=conf.mem_loss_type)
    else:
        mem_model = NoMemory()

    # MODEL

    if conf.model == 'world':
        model: WorldModel = WorldModel(
            encoder=encoder,
            decoder=decoder,
            map_model=map_model,
            mem_model=mem_model,
            action_dim=conf.action_dim,
            deter_dim=conf.deter_dim,
            stoch_dim=conf.stoch_dim,
            hidden_dim=conf.hidden_dim,
            kl_weight=conf.kl_weight,
            map_grad=conf.map_grad,
            embed_rnn=conf.embed_rnn != 'none',
            gru_layers=conf.gru_layers
        )
    elif conf.model == 'map_rnn':
        model = MapPredictModel(
            encoder=encoder,
            decoder=decoder,
            map_model=map_model,
            action_dim=conf.action_dim,
            state_dim=state_dim,
        )  # type: ignore
    else:
        assert False, conf.model

    model.to(device)

    print(f'Model: {param_count(model)} parameters')
    for submodel in [model._encoder, model._decoder_image, model._core, model._input_rnn, map_model, mem_model]:
        if submodel is not None:
            print(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
    # print(model)
    mlflow.set_tag(mlflow.utils.mlflow_tags.MLFLOW_RUN_NOTE, f'```\n{model}\n```')  # type: ignore

    # Training

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore

    resume_step = tools.mlflow_load_checkpoint(model, optimizer)
    if resume_step:
        print(f'Loaded model from checkpoint epoch {resume_step}')

    start_time = time.time()
    steps = resume_step or 0
    last_time = start_time
    last_steps = steps
    metrics = defaultdict(list)
    metrics_max = defaultdict(list)

    state = None

    with get_profiler(conf) as profiler:
        for batch in data.iterate(conf.batch_length, conf.batch_size):
            profiler.step()

            image, action, reset, map = preprocess(batch)
            if state is None or not conf.keep_state:
                state = model.init_state(image.size(1) * conf.iwae_samples)

            # Predict

            output = model.forward(image, action, reset, map, state, I=conf.iwae_samples)  # type: ignore
            state = output[-1]

            # Loss

            loss, loss_metrics, loss_tensors = model.loss(*output, image, map, reset)  # type: ignore

            # Grad step

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
            # var_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters() if p.requires_grad]))
            # metrics['var_norm'].append(var_norm.item())
            optimizer.step()

            # Metrics

            steps += 1
            for k, v in loss_metrics.items():
                metrics[k].append(v.item())
            if grad_norm is not None:
                metrics['grad_norm'].append(grad_norm.item())
                metrics_max['grad_norm_max'].append(grad_norm.item())

            # Log sample

            if steps % conf.log_interval == 1:
                # if (steps == 1) or (steps > 1000 and loss_metrics['loss_model_image_max'].item() > 200):  # DEBUG high loss
                # print(f"[{steps}] Saving batch sample:"
                #       f"  loss_model_image_max: {loss_metrics['loss_model_image_max'].item():.1f}"
                #       f"  loss_model_kl_max: {loss_metrics['loss_model_kl_max'].item():.1f}")
                with torch.no_grad():
                    image_pred, image_rec, map_rec = model.predict(*output)
                log_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec, f'{steps:07}.npz')

            # Log metrics

            if steps % conf.log_interval == 0:
                metrics = {k: np.mean(v) for k, v in metrics.items()}
                metrics.update({k: np.max(v) for k, v in metrics_max.items()})
                metrics['_step'] = steps
                metrics['_loss'] = metrics['loss']

                t = time.time()
                fps = (steps - last_steps) / (t - last_time)
                metrics['fps'] = fps
                last_time, last_steps = t, steps

                print(f"T:{t-start_time:05.0f}  "
                      f"[{steps:06}]"
                      f"  loss_model: {metrics.get('loss_model', 0):.3f}"
                      f"  loss_model_kl: {metrics.get('loss_model_kl', 0):.3f}"
                      f"  loss_model_image: {metrics.get('loss_model_image', 0):.3f}"
                      f"  loss_map: {metrics['loss_map']:.3f}"
                      f"  entropy_prior: {metrics.get('entropy_prior',0):.3f}"
                      f"  entropy_prior_start: {metrics.get('entropy_prior_start',0):.3f}"
                      f"  fps: {metrics['fps']:.3f}"
                      )
                mlflow.log_metrics(metrics, step=steps)
                metrics = defaultdict(list)
                metrics_max = defaultdict(list)

            # Save model

            if steps % conf.save_interval == 0:
                tools.mlflow_save_checkpoint(model, optimizer, steps)
                print(f'Saved model checkpoint')

            # Stop

            if steps >= conf.n_steps:
                print('Stopping')
                break

            # Evaluate

            if conf.eval_interval and steps % conf.eval_interval == 0:
                # Same batch as train
                eval_iter = data_eval.iterate(conf.batch_length, conf.batch_size, skip_first=False)
                evaluate('eval', steps, model, eval_iter, preprocess, conf.eval_batches, conf.iwae_samples, conf.keep_state)

                # Full episodes
                eval_iter_full = data_eval.iterate(conf.full_eval_length, conf.full_eval_size, skip_first=False)
                evaluate('eval_full', steps, model, eval_iter_full, preprocess, conf.full_eval_batches, conf.full_eval_samples, conf.keep_state)


def evaluate(prefix: str,
             steps: int,
             model: WorldModel,
             data_iterator: Iterator,
             preprocess: MinigridPreprocess,
             eval_batches: int,
             eval_samples: int,
             keep_state: bool):

    start_time = time.time()
    metrics_eval = defaultdict(list)
    state = None
    loss_tensors = None
    npz_datas = []
    n_episodes = 0

    for i_batch in range(eval_batches):
        with torch.no_grad():

            batch = next(data_iterator)
            image, action, reset, map = preprocess(batch)
            if i_batch == 0:
                print(f'Evaluation ({prefix}): batches: {eval_batches},  size(N,B,I): {tuple(image.shape[0:2])+(eval_samples,)}')

            if state is not None:  # Non-first batch

                # Log _last predictions from the last batch of previous episode

                if reset.sum() > 0 and loss_tensors is not None:
                    assert all(reset[0].cpu().numpy()), 'First step should be reset'
                    metrics_eval['logprob_map_last'].append(loss_tensors['loss_map'].mean().item())
                    metrics_eval['logprob_img_last'].append(loss_tensors.get('logprob_img', tensor(0.0)).mean().item())
                    n_episodes += image.shape[1]

                # Forward (prior) & unseen logprob

                if reset.sum() == 0:
                    output = model.forward(0 * image[:5], action[:5], reset[:5], map[:5], state, I=eval_samples, imagine=True, do_image_pred=True)
                    _, _, loss_tensors = model.loss(*output, image[:5], map[:5], reset[:5])  # type: ignore
                    if 'logprob_img' in loss_tensors:
                        metrics_eval['logprob_img_1step'].append(loss_tensors['logprob_img'][0].mean().item())
                        metrics_eval['logprob_img_2step'].append(loss_tensors['logprob_img'][1].mean().item())
                    # image_pred, image_rec, map_rec = model.predict(*output)  # TODO: log 5-step prediction sequence

            # Forward (posterior) & loss

            if state is None or not keep_state:
                state = model.init_state(image.size(1) * eval_samples)
            output = model.forward(image, action, reset, map, state, I=eval_samples, do_image_pred=True)
            state = output[-1]

            _, loss_metrics, loss_tensors = model.loss(*output, image, map, reset)  # type: ignore
            metrics_eval['logprob_map'].append(loss_tensors['loss_map'].mean().item())  # Backwards-compat, same as loss_map
            metrics_eval['logprob_img'].append(loss_tensors.get('logprob_img', tensor(0.0)).mean().item())
            for k, v in loss_metrics.items():
                metrics_eval[k].append(v.item())

            # Log one episode batch

            if n_episodes == 0:
                image_pred, image_rec, map_rec = model.predict(*output)
                npz_datas.append(prepare_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec))

    metrics_eval = {f'{prefix}/{k}': np.mean(v) for k, v in metrics_eval.items()}
    mlflow.log_metrics(metrics_eval, step=steps)

    npz_data = {k: np.concatenate([d[k] for d in npz_datas], 1) for k in npz_datas[0]}
    tools.mlflow_log_npz(npz_data, f'{steps:07}.npz', subdir=f'd2_wm_predict_{prefix}', verbose=True)

    print(f'Evaluation ({prefix}): done in {(time.time()-start_time):.0f} sec, recorded {n_episodes} episodes')


def log_batch_npz(batch,
                  loss_tensors,
                  image_pred: Optional[D.Distribution],
                  image_rec: Optional[D.Distribution],
                  map_rec: Optional[D.Distribution],
                  filename: str,
                  subdir='d2_wm_predict'):
    data = prepare_batch_npz(batch, loss_tensors, image_pred, image_rec, map_rec)
    tools.mlflow_log_npz(data, filename, subdir)


def prepare_batch_npz(batch,
                      loss_tensors,
                      image_pred: Optional[D.Distribution],
                      image_rec: Optional[D.Distribution],
                      map_rec: Optional[D.Distribution]):
    data = batch.copy()
    data.update({k: v.cpu().numpy() for k, v in loss_tensors.items()})
    if image_pred is not None:
        if isinstance(image_pred, D.Categorical):
            data['image_pred_p'] = image_pred.probs.cpu().numpy()
        else:
            data['image_pred'] = ((image_pred.mean.cpu().numpy() + 0.5) * 255.0).clip(0, 255).astype('uint8')
    if image_rec is not None:
        if isinstance(image_rec, D.Categorical):
            data['image_rec_p'] = image_rec.probs.cpu().numpy()
        else:
            data['image_rec'] = ((image_rec.mean.cpu().numpy() + 0.5) * 255.0).clip(0, 255).astype('uint8')
    if map_rec is not None:
        if isinstance(map_rec, D.Categorical):
            data['map_rec_p'] = map_rec.probs.cpu().numpy()
        else:
            data['map_rec'] = ((map_rec.mean.cpu().numpy() + 0.5) * 255.0).clip(0, 255).astype('uint8')
    data = {k: v.swapaxes(0, 1) for k, v in data.items()}  # (N,B,...) => (B,N,...)
    return data


def run_generator(conf):
    # Start train
    cmd = f'python3 generator.py {conf.generator_env} --num_steps 1000000000 --seed 1 --output_dir {conf.input_dir} --delete_old {conf.generator_buffer}'
    print(f'Starting data generator:\n{cmd}')
    p1 = subprocess.Popen(cmd.split(' '), stdout=subprocess.DEVNULL)
    # Start eval
    cmd = f'python3 generator.py {conf.generator_env} --num_steps 1000000000 --seed 2 --output_dir {conf.eval_dir} --delete_old {conf.generator_buffer} --sleep 20'
    print(f'Starting data generator:\n{cmd}')
    p2 = subprocess.Popen(cmd.split(' '), stdout=subprocess.DEVNULL)
    # Check
    time.sleep(5)
    assert (p1.poll() is None) and (p2.poll() is None), 'Process has exited'
    # Wait
    print(f'Waiting for {conf.generator_wait} sec for initial data')
    time.sleep(conf.generator_wait)
    # Check again
    assert (p1.poll() is None) and (p2.poll() is None), 'Process has exited'


def get_profiler(conf):
    if conf.enable_profiler:
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=2, repeat=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        )
    else:
        return NoProfiler()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    for name in args.configs:
        conf.update(configs[name])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    conf = parser.parse_args(remaining)

    run(conf)
