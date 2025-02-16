import json
import time
import torch
import os
import threading

class Profile:
    def __init__(self, model, name="model", schedule=None):
        self.name_map = self._build_name_map(model, name)

        self.events = []  # collected trace events
        self.hooks = []  # hooks for removal
        self.fwd_start = {}
        self.bwd_start = {}
        self.current_step = 0
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

    # ---------------- Forward Hooks ----------------
    def _forward_pre_hook(self, module, inputs):
        if self.get_phase() == "active":
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("Forward: " + self.name_map[module])
            self.fwd_start[module] = time.perf_counter()

    def _forward_post_hook(self, module, inputs, outputs):
        if self.get_phase() == "active":
            start_time = self.fwd_start.pop(module, None)
            if start_time is not None:
                duration = time.perf_counter() - start_time
                event = {
                    "name": self.name_map[module],
                    "phase": "forward",
                    "ts": start_time * 1e6,  # timestamps in microseconds
                    "dur": duration * 1e6,
                    "pid": os.getpid(),
                    "tid": threading.get_ident(),
                }
                self.events.append(event)

            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

        if torch.is_tensor(outputs) and outputs.requires_grad:
            outputs.register_hook(lambda grad: self._tensor_backward_hook(module, grad))

    # ---------------- Backward Hooks ----------------
    def _backward_pre_hook(self, module, grad_output):
        # Record the start time of the backward pass for this module.
        self.bwd_start[module] = time.perf_counter()

    def _backward_post_hook(self, module, grad_input, grad_output):
        start_time = self.bwd_start.pop(module, None)
        if start_time is not None and self.get_phase() == "active":
            duration = time.perf_counter() - start_time
            event = {
                "name": self.name_map[module],
                "phase": "backward",
                "ts": start_time * 1e6,
                "dur": duration * 1e6,
                "pid": os.getpid(),
                "tid": threading.get_ident(),
            }
            self.events.append(event)
    
    def _backward_hook(self, module, grad_input, grad_output):
        if self.get_phase() == "active" and torch.cuda.is_available():
            torch.cuda.nvtx.range_push("Backward: " + self.name_map[module])
        self._backward_pre_hook(module, grad_output)
        self._backward_post_hook(module, grad_input, grad_output)
        if self.get_phase() == "active" and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
    
    def _tensor_backward_hook(self, module, grad):
        if self.get_phase() == "active":
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("Backward Tensor: " + self.name_map[module])
            t_start = time.perf_counter()
            t_end = time.perf_counter()
            duration = t_end - t_start
            event = {
                "name": self.name_map[module],
                "phase": "backward_tensor",
                "ts": t_start * 1e6,
                "dur": duration * 1e6,
            }
            self.events.append(event)
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        return grad

    # ---------------- Context Manager ----------------
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
        self.current_step += 1

    def get_phase(self):
        total_wait = self.schedule.get("wait", 0)
        total_warmup = self.schedule.get("warmup", 0)
        if self.current_step < total_wait:
            return "wait"
        elif self.current_step < total_wait + total_warmup:
            return "warmup"
        else:
            return "active"

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        trace_events = []
        for event in self.events:
            trace_event = {
                "name": event["name"],
                "cat": event["phase"],
                "ph": "X",  # "X" means a complete event (with duration)
                "ts": event["ts"],
                "dur": event["dur"],
                "pid": event.get("pid", "unknown"),
                "tid": event.get("tid", "unknown"),
            }
            trace_events.append(trace_event)
        trace_dict = {"traceEvents": trace_events}
        with open(path, "w") as f:
            json.dump(trace_dict, f, indent=2)
        print(f"Trace written to {path}")
