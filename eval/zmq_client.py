#!/usr/bin/env python3
"""Minimal, dependency-light client for GR00T's ZMQ inference server (scripts/inference_service.py --server).

Re-implements the wire format of gr00t/eval/service.py (msgpack + npy-encoded ndarrays) so the simulator venvs
(LIBERO / RoboCasa, different numpy/robosuite pins) do not need to install gr00t or torch.
Needs only: numpy, pyzmq, msgpack.

CLI:  python eval/zmq_client.py --port 5555 --wait 600     # block until the server answers a ping
"""
import argparse
import io
import sys
import time

import msgpack
import numpy as np
import zmq


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _decode(obj):
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class PolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=60_000):
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=_encode))
        try:
            resp = msgpack.unpackb(self.sock.recv(), object_hook=_decode)
        except zmq.Again:
            self.sock.close()
            self._connect()  # REQ socket is stuck after a timeout
            raise TimeoutError(f"no reply from policy server on port {self.port}")
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def ping(self):
        return self.call("ping", requires_input=False)

    def get_action(self, obs: dict) -> dict:
        """obs: unbatched, e.g. video.* (T,H,W,C) uint8, state.* (T,D), annotation.* [str]. Returns action.* (H,D)."""
        return self.call("get_action", obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for the server")
    a = ap.parse_args()
    t0 = time.time()
    while True:
        try:
            c = PolicyClient(a.host, a.port, timeout_ms=3000)
            print("server up:", c.ping())
            return 0
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > a.wait:
                print("server not reachable:", e, file=sys.stderr)
                return 1
            time.sleep(3)


if __name__ == "__main__":
    sys.exit(main())

