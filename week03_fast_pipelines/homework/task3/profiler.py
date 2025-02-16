import json
import time
import torch
import os
import threading
from collections import defaultdict

# -----------------------------
# Custom Profiler Implementation
# -----------------------------
class Profile:
    def __init__(self, model, name="model", schedule=None):
        self.name_map = self._build_name_map(model, name)

        self.events = []  # collected trace events
        self.hooks = []  # hooks for removal
        self.fwd_start_times = {}
        self.bwd_start_times = {}
        self.step_count = 0
        # Schedule: dictionary with keys "wait", "warmup", "active"
        self.schedule = schedule or {"wait": 1, "warmup": 1, "active": 3}


    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        if self.get_phase() != "active":
            return
        self.fwd_start_times[module] = time.perf_counter()

    def _forward_post_hook(self, module, inputs, outputs):
        if self.get_phase() != "active":
            return
        start_time = self.fwd_start_times.pop(module, None)
        if start_time is None:
            return
        duration = time.perf_counter() - start_time
        event = {
            "name": self.name_map[module],
            "phase": "forward",
            "ts": start_time * 1e6,  # timestamps in microseconds
            "dur": duration * 1e6,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "args": {}
        }
        self.events.append(event)


    def _backward_pre_hook(self, module, grad_output):
        # Record the start time of the backward pass for this module.
        self.bwd_start_times[module] = time.perf_counter()

    def _backward_post_hook(self, module, grad_input, grad_output):
        start_time = self.bwd_start_times.pop(module, None)
        if start_time is None:
            return
        duration = time.perf_counter() - start_time
        event = {
            "name": self.name_map[module],
            "phase": "backward",
            "ts": start_time * 1e6,
            "dur": duration * 1e6,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "args": {}
        }
        self.events.append(event)
    
    def _backward_hook(self, module, grad_input, grad_output):
        # Since PyTorch does not provide separate backward pre and post hooks,
        # we simulate them by recording a timestamp at the start and then immediately
        # recording the duration. (In practice, this may only capture a portion of the
        # backward computation time.)
        self._backward_pre_hook(module, grad_output)
        self._backward_post_hook(module, grad_input, grad_output)

    def __enter__(self):
        # Register forward pre and post hooks on every leaf module.
        for module in self.name_map.keys():
            if self._is_leaf(module):
                self.hooks.append(module.register_forward_pre_hook(self._forward_pre_hook))
                self.hooks.append(module.register_forward_hook(self._forward_post_hook))
                self.hooks.append(module.register_full_backward_hook(self._backward_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Remove all registered hooks.
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def step(self):
        self.step_count += 1

    def get_phase(self):
        total_wait = self.schedule.get("wait", 0)
        total_warmup = self.schedule.get("warmup", 0)
        if self.step_count < total_wait:
            return "wait"
        elif self.step_count < total_wait + total_warmup:
            return "warmup"
        else:
            return "active"

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        trace_dict = {"traceEvents": self.events}
        with open(path, "w") as f:
            json.dump(trace_dict, f, indent=2)
        print(f"Trace written to {path}")
