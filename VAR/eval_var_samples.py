import argparse
import io
import json
import math
import os
import random
import sys
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg
from tqdm.auto import tqdm

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception as exc:  # pragma: no cover - exercised by shell dependency checks.
    tf = None
    TF_IMPORT_ERROR = exc
else:
    TF_IMPORT_ERROR = None


FID_POOL_NAME = 'pool_3:0'
FID_SPATIAL_NAME = 'mixed_6/conv:0'
DEFAULT_REFERENCE_NPZ = Path('evaluation/VIRTUAL_imagenet256_labeled.npz')


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        if mu1.shape != mu2.shape:
            raise ValueError(f'mean vectors have different lengths: {mu1.shape} vs {mu2.shape}')
        if sigma1.shape != sigma2.shape:
            raise ValueError(f'covariances have different dimensions: {sigma1.shape} vs {sigma2.shape}')

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f'FID covmean has imaginary component {np.max(np.abs(covmean.imag))}')
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def resolve_reference_npz(path: str) -> str:
    if path:
        return path
    return str(DEFAULT_REFERENCE_NPZ)


def list_pngs(sample_dir: Path):
    pngs = sorted(
        [p for p in sample_dir.iterdir() if p.suffix.lower() == '.png'],
        key=lambda p: p.name,
    )
    return pngs


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int):
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    @abstractmethod
    def shape(self):
        pass

    def read_batches(self, batch_size: int):
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self._shape = tuple(shape)
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int):
        if self.idx >= self._shape[0]:
            return None
        bs = min(batch_size, self._shape[0] - self.idx)
        self.idx += bs
        read_count = bs * np.prod(self._shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, 'array data')
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self._shape[1:]])

    def remaining(self) -> int:
        return max(0, self._shape[0] - self.idx)

    def shape(self):
        return self._shape


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: Path, arr_name: str):
        with open(path, 'rb') as fp:
            arr = np.load(fp)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int):
        if self.idx >= self.arr.shape[0]:
            return None
        res = self.arr[self.idx:self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)

    def shape(self):
        return tuple(self.arr.shape)


@contextmanager
def open_npz_array(path: Path, arr_name: str):
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


@contextmanager
def _open_npy_file(path: Path, arr_name: str):
    with open(path, 'rb') as fp:
        with zipfile.ZipFile(fp, 'r') as zip_f:
            if f'{arr_name}.npy' not in zip_f.namelist():
                raise ValueError(f'missing {arr_name} in npz file: {path}')
            with zip_f.open(f'{arr_name}.npy', 'r') as arr_f:
                yield arr_f


def _read_bytes(fp, size, error_template='ran out of data'):
    data = bytes()
    while True:
        try:
            chunk = fp.read(size - len(data))
            data += chunk
            if len(chunk) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        raise ValueError(f'EOF: reading {error_template}, expected {size} bytes got {len(data)}')
    return data


def image_batch_iter(paths, batch_size: int, resolution: int):
    for begin in range(0, len(paths), batch_size):
        batch_paths = paths[begin:begin + batch_size]
        arrays = []
        for path in batch_paths:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if img.size != (resolution, resolution):
                    raise ValueError(f'{path} has size {img.size}, expected {(resolution, resolution)}')
                arrays.append(np.asarray(img, dtype=np.uint8))
        yield np.stack(arrays, axis=0)


def validate_image_batch(batch: np.ndarray, resolution: int, source: str):
    if batch.dtype != np.uint8:
        raise ValueError(f'{source} has dtype={batch.dtype}, expected uint8')
    if batch.ndim != 4 or batch.shape[1:] != (resolution, resolution, 3):
        raise ValueError(f'{source} has shape={batch.shape}, expected [N,{resolution},{resolution},3]')


def build_npz_from_pngs(paths, npz_path: Path, batch_size: int, resolution: int, overwrite: bool = False):
    if npz_path.exists() and not overwrite:
        with open_npz_array(npz_path, 'arr_0') as reader:
            shape = reader.shape()
        if shape == (len(paths), resolution, resolution, 3):
            print(f'[NPZ] reuse existing {npz_path} [shape={shape}]')
            return npz_path
        raise ValueError(
            f'existing sample npz has shape={shape}, expected {(len(paths), resolution, resolution, 3)}; '
            'rerun with --overwrite_npz 1 to rebuild it'
        )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = npz_path.with_name(npz_path.name + '.arr_0.npy.tmp')
    arr = np.lib.format.open_memmap(
        tmp_npy,
        mode='w+',
        dtype=np.uint8,
        shape=(len(paths), resolution, resolution, 3),
    )
    offset = 0
    for batch in tqdm(image_batch_iter(paths, batch_size, resolution), total=math.ceil(len(paths) / batch_size), desc='Building sample npz'):
        validate_image_batch(batch, resolution, 'png batch')
        arr[offset:offset + batch.shape[0]] = batch
        offset += batch.shape[0]
    arr.flush()
    del arr

    tmp_npz = npz_path.with_name(npz_path.name + '.tmp')
    with zipfile.ZipFile(tmp_npz, mode='w', compression=zipfile.ZIP_STORED) as zip_f:
        zip_f.write(tmp_npy, arcname='arr_0.npy')
    tmp_npz.replace(npz_path)
    tmp_npy.unlink()
    print(f'[NPZ] saved {npz_path} [shape={(len(paths), resolution, resolution, 3)}]')
    return npz_path


def _update_shapes(pool3):
    ops = pool3.graph.get_operations()
    for op in ops:
        for output in op.outputs:
            shape = output.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                new_shape = []
                for idx, dim in enumerate(shape):
                    new_shape.append(None if dim == 1 and idx == 0 else dim)
                output.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


def create_feature_graph(input_batch, inception_pb: Path):
    prefix = f'{random.randrange(2**32)}_{random.randrange(2**32)}'
    with open(inception_pb, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={'ExpandDims:0': input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def create_softmax_graph(input_batch, inception_pb: Path):
    prefix = f'{random.randrange(2**32)}_{random.randrange(2**32)}'
    with open(inception_pb, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
    (matmul,) = tf.import_graph_def(
        graph_def,
        return_elements=['softmax/logits/MatMul'],
        name=prefix,
    )
    logits = tf.matmul(input_batch, matmul.inputs[1])
    return tf.nn.softmax(logits)


def compute_activations(session, image_input, pool_features, spatial_features, batches, num_batches):
    preds = []
    spatial_preds = []
    for batch in tqdm(batches, total=num_batches, desc='Inception activations'):
        batch = batch.astype(np.float32)
        pred, spatial_pred = session.run(
            [pool_features, spatial_features],
            {image_input: batch},
        )
        preds.append(pred.reshape([pred.shape[0], -1]))
        spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
    return np.concatenate(preds, axis=0), np.concatenate(spatial_preds, axis=0)


def compute_statistics(activations: np.ndarray):
    return FIDStatistics(np.mean(activations, axis=0), np.cov(activations, rowvar=False))


def compute_inception_score(session, softmax_input, softmax, activations: np.ndarray, split_size: int = 5000, softmax_batch_size: int = 512):
    softmax_out = []
    for begin in range(0, len(activations), softmax_batch_size):
        acts = activations[begin:begin + softmax_batch_size]
        softmax_out.append(session.run(softmax, feed_dict={softmax_input: acts}))
    preds = np.concatenate(softmax_out, axis=0)
    scores = []
    split_size = max(1, min(int(split_size), len(preds)))
    for begin in range(0, len(preds), split_size):
        part = preds[begin:begin + split_size]
        kl = part * (np.log(part + 1e-20) - np.log(np.expand_dims(np.mean(part, 0), 0) + 1e-20))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def load_reference_stats(path: Path):
    obj = np.load(path)
    required = ['mu', 'sigma', 'mu_s', 'sigma_s']
    missing = [name for name in required if name not in obj.files]
    if missing:
        raise ValueError(f'{path} is missing precomputed statistics: {missing}')
    return FIDStatistics(obj['mu'], obj['sigma']), FIDStatistics(obj['mu_s'], obj['sigma_s'])


def load_sample_metadata(sample_dir: Path):
    metadata_path = sample_dir / 'metadata.json'
    if not metadata_path.is_file():
        return None
    with open(metadata_path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def validate_official_imagenet256(args, sample_dir: Path, sample_count: int, metadata):
    if not args.require_official_imagenet256:
        return False

    if args.expected_count != 50000:
        raise ValueError(f'official ImageNet256 FID/IS expects --expected_count 50000, got {args.expected_count}')
    if args.resolution != 256:
        raise ValueError(f'official ImageNet256 FID/IS expects --resolution 256, got {args.resolution}')
    if sample_count != 50000:
        raise ValueError(f'official ImageNet256 FID/IS expects 50,000 samples, got {sample_count}')
    if metadata is None:
        raise ValueError(f'official evaluation requires {sample_dir / "metadata.json"}')
    if not metadata.get('official_imagenet256', False):
        raise ValueError('metadata.json does not mark this folder as official_imagenet256')

    meta_args = metadata.get('args', {})
    expected = {
        'label_mode': 'balanced',
        'samples_per_class': 50,
        'class_start': 0,
        'class_end': 999,
        'num_classes': 1000,
        'num_images': 50000,
        'cfg': 1.5,
        'top_k': 900,
        'top_p': 0.96,
        'more_smooth': 0,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual = meta_args.get(key)
        if isinstance(expected_value, float):
            ok = abs(float(actual) - expected_value) < 1e-12
        else:
            ok = actual == expected_value
        if not ok:
            mismatches.append((key, actual, expected_value))
    if mismatches:
        raise ValueError(f'official metadata mismatch: {mismatches}')

    hist = metadata.get('label_histogram', {})
    if len(hist) != 1000 or any(int(hist.get(str(class_id), -1)) != 50 for class_id in range(1000)):
        raise ValueError('official metadata must contain exactly 50 samples for every ImageNet class 0..999')
    return True


def write_result(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        for key in [
            'status',
            'metric_protocol',
            'official_imagenet256',
            'sample_count',
            'fid',
            'sfid',
            'inception_score',
            'inception_score_std',
            'error',
        ]:
            if key in payload:
                fp.write(f'{key}: {payload[key]}\n')
        fp.write('\n')
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write('\n')


def build_argparser():
    parser = argparse.ArgumentParser(description='Evaluate VAR PNG samples with OpenAI Inception FID/IS metrics.')
    parser.add_argument('--sample_dir', type=str, required=True)
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--reference_npz', type=str, default='')
    parser.add_argument('--inception_pb', type=str, default='classify_image_graph_def.pb')
    parser.add_argument('--expected_count', type=int, default=50000)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--softmax_batch_size', type=int, default=512)
    parser.add_argument('--is_split_size', type=int, default=5000)
    parser.add_argument('--metric_mode', type=str, default='official_imagenet256', choices=['official_imagenet256'])
    parser.add_argument('--require_official_imagenet256', type=int, default=1, choices=[0, 1])
    parser.add_argument('--sample_npz', type=str, default='')
    parser.add_argument('--save_npz', type=int, default=1, choices=[0, 1])
    parser.add_argument('--overwrite_npz', type=int, default=0, choices=[0, 1])
    return parser


def main():
    args = build_argparser().parse_args()
    sample_dir = Path(args.sample_dir)
    result_file = Path(args.result_file)
    reference_npz = Path(resolve_reference_npz(args.reference_npz))
    inception_pb = Path(args.inception_pb)
    sample_npz = Path(args.sample_npz) if args.sample_npz else Path(f'{sample_dir}.npz')

    try:
        if tf is None:
            raise RuntimeError(f'tensorflow import failed: {TF_IMPORT_ERROR}')
        if not sample_dir.is_dir():
            raise FileNotFoundError(f'sample_dir is missing: {sample_dir}')
        if not reference_npz.is_file():
            raise FileNotFoundError(f'reference_npz is missing: {reference_npz}')
        if not inception_pb.is_file():
            raise FileNotFoundError(f'inception_pb is missing: {inception_pb}')

        pngs = list_pngs(sample_dir)
        sample_count = len(pngs)
        if args.expected_count > 0 and sample_count != args.expected_count:
            raise ValueError(f'{sample_count} png files found in {sample_dir}, expected {args.expected_count}')
        if sample_count == 0:
            raise ValueError(f'no png files found in {sample_dir}')
        metadata = load_sample_metadata(sample_dir)
        official_ok = validate_official_imagenet256(args, sample_dir, sample_count, metadata)

        input_source = 'png'
        if args.save_npz:
            sample_npz = build_npz_from_pngs(
                pngs,
                sample_npz,
                batch_size=max(1, args.batch_size),
                resolution=args.resolution,
                overwrite=bool(args.overwrite_npz),
            )
            input_source = 'npz'

        ref_stats, ref_stats_spatial = load_reference_stats(reference_npz)
        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            pool_features, spatial_features = create_feature_graph(image_input, inception_pb)
            softmax = create_softmax_graph(softmax_input, inception_pb)
            session.run(tf.global_variables_initializer())
            session.run([pool_features, spatial_features], {image_input: np.zeros([1, 64, 64, 3], dtype=np.float32)})

            if input_source == 'npz':
                with open_npz_array(sample_npz, 'arr_0') as reader:
                    shape = reader.shape()
                    if shape != (sample_count, args.resolution, args.resolution, 3):
                        raise ValueError(
                            f'{sample_npz} has shape={shape}, expected '
                            f'{(sample_count, args.resolution, args.resolution, 3)}'
                        )
                    batches = reader.read_batches(args.batch_size)
                    sample_pool, sample_spatial = compute_activations(
                        session,
                        image_input,
                        pool_features,
                        spatial_features,
                        batches,
                        len(batches),
                    )
            else:
                num_batches = int(math.ceil(sample_count / max(1, args.batch_size)))
                batches = image_batch_iter(pngs, args.batch_size, args.resolution)
                sample_pool, sample_spatial = compute_activations(
                    session,
                    image_input,
                    pool_features,
                    spatial_features,
                    batches,
                    num_batches,
                )
            sample_stats = compute_statistics(sample_pool)
            sample_stats_spatial = compute_statistics(sample_spatial)
            is_mean, is_std = compute_inception_score(
                session,
                softmax_input,
                softmax,
                sample_pool,
                split_size=args.is_split_size,
                softmax_batch_size=args.softmax_batch_size,
            )

        payload = {
            'status': 'ok',
            'metric_protocol': args.metric_mode,
            'official_imagenet256': bool(official_ok),
            'sample_dir': str(sample_dir),
            'sample_npz': str(sample_npz) if input_source == 'npz' else None,
            'input_source': input_source,
            'sample_count': sample_count,
            'reference_npz': str(reference_npz),
            'inception_pb': str(inception_pb),
            'batch_size': int(args.batch_size),
            'softmax_batch_size': int(args.softmax_batch_size),
            'is_split_size': int(args.is_split_size),
            'fid': sample_stats.frechet_distance(ref_stats),
            'sfid': sample_stats_spatial.frechet_distance(ref_stats_spatial),
            'inception_score': is_mean,
            'inception_score_std': is_std,
        }
        write_result(result_file, payload)
        print(f"FID: {payload['fid']}")
        print(f"sFID: {payload['sfid']}")
        print(f"IS: {payload['inception_score']} +/- {payload['inception_score_std']}")
        print(f'Result saved to {result_file}')
    except Exception as exc:
        payload = {
            'status': 'failed',
            'metric_protocol': args.metric_mode,
            'official_imagenet256': False,
            'sample_dir': str(sample_dir),
            'sample_count': len(list_pngs(sample_dir)) if sample_dir.is_dir() else 0,
            'reference_npz': str(reference_npz),
            'inception_pb': str(inception_pb),
            'sample_npz': str(sample_npz),
            'error': f'{type(exc).__name__}: {exc}',
        }
        write_result(result_file, payload)
        print(payload['error'], file=sys.stderr)
        print(f'Result saved to {result_file}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
