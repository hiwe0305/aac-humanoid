"""Verify metadata and, optionally, parquet and video content for LeRobot v2 datasets."""
import json
from pathlib import Path


def verify_dataset(path: Path, deep=False, expected_cameras=None):
    meta = path / 'meta'
    required = [meta / 'info.json', meta / 'episodes.jsonl', meta / 'tasks.jsonl', meta / 'modality.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('missing metadata: ' + ', '.join(missing))
    info = json.loads((meta / 'info.json').read_text())
    episodes = [json.loads(line) for line in (meta / 'episodes.jsonl').read_text().splitlines() if line.strip()]
    if len(episodes) != info['total_episodes'] or sum(ep['length'] for ep in episodes) != info['total_frames']:
        raise ValueError('episode/frame counts do not match info.json')
    video_keys = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
    if not episodes or not video_keys or (expected_cameras is not None and len(video_keys) != expected_cameras):
        raise ValueError(f'expected {expected_cameras} camera streams and nonempty episodes; got {len(video_keys)} cameras')
    if deep:
        import av
        import pyarrow.parquet as pq
        import numpy as np
    seen = set()
    for ep in episodes:
        index = ep['episode_index']
        if index in seen:
            raise ValueError(f'duplicate episode index {index}')
        seen.add(index)
        fields = {'episode_index': index, 'episode_chunk': index // info['chunks_size']}
        parquet = path / info['data_path'].format(**fields)
        videos = [path / info['video_path'].format(video_key=key, **fields) for key in video_keys]
        for file in [parquet, *videos]:
            if not file.is_file() or not file.stat().st_size:
                raise ValueError(f'missing or empty file: {file}')
        if deep:
            parquet_file = pq.ParquetFile(parquet)
            if parquet_file.metadata.num_rows != ep['length']:
                raise ValueError(f'parquet rows differ from episode length: {parquet}')
            missing_columns = {'action', 'observation.state'} - set(parquet_file.schema_arrow.names)
            if missing_columns:
                raise ValueError(f'missing parquet columns {sorted(missing_columns)}: {parquet}')
            table = parquet_file.read(columns=['action', 'observation.state'])
            for key in ('action', 'observation.state'):
                values = np.asarray(table[key].to_pylist(), dtype=np.float64)
                expected_shape = (ep['length'], *info['features'][key]['shape'])
                if values.shape != expected_shape or not np.isfinite(values).all():
                    raise ValueError(f'invalid shape or non-finite values in {key}: {parquet}')
            for key, video in zip(video_keys, videos):
                with av.open(str(video)) as container:
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise ValueError(f'cannot decode first video frame: {video}')
                    height, width = info['features'][key]['shape'][:2]
                    if (frame.height, frame.width) != (height, width):
                        raise ValueError(f'video size does not match metadata: {video}')
    return len(episodes), info['total_frames'], len(video_keys)
