import pathlib
import tempfile
import time

from torch._C._distributed_c10d import _register_handler, _Request, _Response
from torch.profiler import _ExperimentalConfig, profile


def _torch_profile(req: _Request, resp: _Response) -> None:
    experimental_config = _ExperimentalConfig(
        profile_all_threads=True,
    )
    duration = float(req.get_param("duration"))
    with profile(record_shapes=True, experimental_config=experimental_config) as prof:
        time.sleep(duration)

    with tempfile.NamedTemporaryFile(prefix="torch_debug", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        resp.set_content(pathlib.Path(f.name).read_bytes(), "application/json")
        resp.set_status(200)


_register_handler("torch_profile", _torch_profile)


# Pickle-based NCCL trace handler that includes stack frames
# The C++ dump_nccl_trace_json endpoint strips frames from JSON output;
# this uses _dump_nccl_trace() (pickle bytes) which includes them.
def _dump_nccl_trace_with_frames(req: _Request, resp: _Response) -> None:
    import json
    import pickle

    import torch

    try:
        trace_bytes = torch._C._distributed_c10d._dump_nccl_trace()
        trace_data = pickle.loads(trace_bytes)

        if isinstance(trace_data, dict):
            for pg_key, pg_data in trace_data.items():
                entries = []
                if isinstance(pg_data, list):
                    entries = pg_data
                elif isinstance(pg_data, dict):
                    entries = pg_data.get("entries", [])
                for entry in entries:
                    if "frames" in entry and isinstance(entry["frames"], list):
                        formatted = []
                        for f in entry["frames"]:
                            if isinstance(f, dict):
                                formatted.append(f)
                            elif isinstance(f, (list, tuple)) and len(f) >= 3:
                                formatted.append(
                                    {
                                        "name": str(f[0]),
                                        "filename": str(f[1]),
                                        "line": int(f[2]),
                                    }
                                )
                        entry["frames"] = formatted

        resp.set_content(json.dumps(trace_data).encode(), "application/json")
        resp.set_status(200)
    except Exception as e:
        resp.set_content(json.dumps({"error": str(e)}).encode(), "application/json")
        resp.set_status(500)


_register_handler("dump_nccl_trace_with_frames", _dump_nccl_trace_with_frames)
