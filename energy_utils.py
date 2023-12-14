import time

import pandas as pd
from multiprocessing import Process, Queue, Event
#import os
#import subprocess
#import io

def get_energy_profiler(hardware):
    if hardware == "ipu":
        return GetIPUPower
    elif hardware == 'gpu':
        return GetNVIDIAPower
    elif hardware == 'amd':
        return GetARMPower
    else:
        raise NotImplementedError(f"Unknown hardware {name}")

#NVIDIA GPUS
class GetNVIDIAPower(object):
    
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()
        
        interval = 100 #ms
        self.smip = Process(target=self._power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self
    
    def _power_loop(self,queue, event, interval):
        import pynvml as pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        device_list = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(device_count)]
        power_value_dict = {
            idx : [] for idx in range(device_count)
        }
        power_value_dict['timestamps'] = []
        last_timestamp = time.time()

        while not event.is_set():
            for idx,handle in enumerate(device_list):
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_value_dict[idx].append(power*1e-3)
            timestamp = time.time()
            power_value_dict['timestamps'].append(timestamp)
            wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
            time.sleep(wait_for)
            last_timestamp = timestamp
        queue.put(power_value_dict)

    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)
        
    def energy(self):
        import numpy as np
        _energy = []
        energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy

    
#ARM GPUS


class GetARMPower(object):
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()
        
        interval = 100 #ms
        self.smip = Process(target=self._power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self
    
    def _power_loop(self,queue, event, interval):
        import rsmiBindings as rsmiBindings
        ret = rsmiBindings.rocmsmi.rsmi_init(0)
        if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError("Failed initializing rocm_smi library")
        device_count = c_uint32(0)
        ret = rsmiBindings.rocmsmi.rsmi_num_monitor_devices(byref(device_count))
        if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError("Failed enumerating ROCm devices")
        device_list = list(range(device_count.value))
        power_value_dict = {
            id : [] for id in device_list
        }
        power_value_dict['timestamps'] = []
        last_timestamp = time.time()
        start_energy_list = []
        for id in device_list:
            energy = c_uint64()
            energy_timestamp = c_uint64()
            energy_resolution = c_float()
            ret = rsmiBindings.rocmsmi.rsmi_dev_energy_count_get(id, 
                    byref(energy),
                    byref(energy_resolution),
                    byref(energy_timestamp))
            if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                raise RuntimeError(f"Failed getting Power of device {id}")
            start_energy_list.append(round(energy.value*energy_resolution.value,2)) # unit is uJ

        while not event.is_set():
            for id in device_list:
                power = c_uint32()
                ret = rsmiBindings.rocmsmi.rsmi_dev_power_ave_get(id, 0, byref(power))
                if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                    raise RuntimeError(f"Failed getting Power of device {id}")
                power_value_dict[id].append(power.value*1e-6) # value is uW
            timestamp = time.time()
            power_value_dict['timestamps'].append(timestamp)
            wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
            time.sleep(wait_for)
            last_timestamp = timestamp

        energy_list = [0.0 for _ in device_list]
        for id in device_list:
            energy = c_uint64()
            energy_timestamp = c_uint64()
            energy_resolution = c_float()
            ret = rsmiBindings.rocmsmi.rsmi_dev_energy_count_get(id, 
                    byref(energy),
                    byref(energy_resolution),
                    byref(energy_timestamp))
            if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                raise RuntimeError(f"Failed getting Power of device {id}")
            energy_list[id] = round(energy.value*energy_resolution.value,2) - start_energy_list[id]

        energy_list = [ (energy*1e-6)/3600 for energy in energy_list] # convert uJ to Wh
        queue.put(power_value_dict)
        queue.put(energy_list)

    
    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.energy_list_counter = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)
    def energy(self):
        import numpy as np
        _energy = []
        energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy,self.energy_list_counter

    
    
    
#IPUS

class GetIPUPower(object):   
    # PLACEHOLDER FOR NOW
    
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()
        
        interval = 100 #ms
        self.smip = Process(target=self._power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self
    
    def _power_loop(self,queue, event, interval):
    
        device_count = 1
        device_list = [0]
        power_value_dict = {
            idx : [] for idx in range(device_count)
        }
        power_value_dict['timestamps'] = []
        last_timestamp = time.time()

        while not event.is_set():
            for idx,handle in enumerate(device_list):
                power = 1
                power_value_dict[idx].append(power*1e-3)
            timestamp = time.time()
            power_value_dict['timestamps'].append(timestamp)
            wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
            time.sleep(wait_for)
            last_timestamp = timestamp
        queue.put(power_value_dict)

    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame()
        
    def energy(self):
        import numpy as np
        _energy = []
        #energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        #_energy = energy_df[1:].sum(axis=0).values.tolist()
        _energy=0
        return _energy

    
    
# import argparse
# import time
# import sys
# import math
# import os

# # Requires asciichartpy: pip3 install --user asciichartpy
# import asciichartpy
# import gcipuinfo

# ipu_info = gcipuinfo.gcipuinfo()
# num_devices = len(ipu_info.getDevices())
# if num_devices == 0:
#     print("gc_power_consumption.py: error: no IPUs detected", file=sys.stderr)
#     exit(-1)


# def get_ipu_power_single(device_id):
#     if 0 <= device_id and device_id < num_devices:
#         return pow_to_float(
#             ipu_info.getNamedAttributeForDevice(device_id, gcipuinfo.IpuPower)
#         )
#     else:
#         print(
#             f"gc_power_consumption.py: error: device id {device_id} does not exist (valid range is 0-{num_devices-1})",
#             file=sys.stderr,
#         )
#         exit(-1)


# def get_ipu_power_from_device_list(devices):
#     powers = []
#     for device_id in devices:
#         pow = ipu_info.getNamedAttributeForDevice(device_id, gcipuinfo.IpuPower)
#         if pow != "N/A":
#             powers.append(pow_to_float(pow))
#     return powers

# def get_ipu_power_all():
#     device_powers = ipu_info.getNamedAttributeForAll(gcipuinfo.IpuPower)
#     return [pow_to_float(pow) for pow in device_powers if pow != "N/A"]


# def pow_to_float(pow):
#     # Power is reported in the format xxx.xxW, so remove the last character.
#     # We also handle the case when the power reports as N/A.
#     try:
#         return float(pow[:-1])
#     except ValueError:
#         return 0


# def draw_graph(power_history, mode, num_devices, device_ids, min, max, width, height):
#     graph_cfg = {
#         "height": height - 3,  # Leave room for the title at the top
#         "format": "{:8.2f}W ",
#         "min": min if min else 0,
#     }
#     if max and max > graph_cfg["min"]:
#         graph_cfg["max"] = max

#     if device_ids:
#         title_str = mode.capitalize() + " power consumption for IPUs: " + ", ".join(map(str, device_ids))
#     else:
#         title_str = mode.capitalize() + " power consumption for " + str(num_devices) + " IPUs"

#     print(title_str.center(width))
#     graph = asciichartpy.plot(power_history, graph_cfg) + "\n"
#     sys.stdout.buffer.write(graph.encode("utf-8"))
#     sys.stdout.flush()


# def main():
#     parser = argparse.ArgumentParser(description="Display a console graph of IPU power consumption over time")
#     parser.add_argument("--min", type=float, help="Minimum y-axis value, in watts")
#     parser.add_argument("--max", type=float, help="Maximum y-axis value, in watts")
#     parser.add_argument(
#         "--interval",
#         type=float,
#         default=1,
#         help="Interval between power queries, in seconds",
#     )
#     parser.add_argument(
#         "--devices", type=int, nargs="+", help="only query specific devices"
#     )
#     parser.add_argument('--mode',  help='Simulator IPU architecture',
#                         choices=["mean", "total"],
#                         required=False, default="mean")

#     # This example assumes per-IPU power sensors, which are not available on
#     # C2 devices
#     if ipu_info.getNamedAttributeForDevice(0, gcipuinfo.BoardType) != "M2000":
#         print("This program is only supported on IPU-Machine devices")
#         sys.exit(1)

#     args = parser.parse_args()

#     try:
#         term_width, term_height = os.get_terminal_size()
#     except OSError:
#         print(
#             "gc_power_consumption.py: warning: stdout is not attached to a tty, using 50x50 graph",
#             file=sys.stderr,
#         )
#         term_width, term_height = 50

#     power_history = []
#     max_entries = term_width - 15  # Leave enough room for the y-axis labels

#     while True:
#         if not args.devices:
#             powers = get_ipu_power_all()
#         else:
#             powers = get_ipu_power_from_device_list(args.devices)
#         if len(powers) > 0:
#             if args.mode == "mean":
#                 val = sum(powers) / len(powers)
#             else:
#                 val = sum(powers)
#             power_history.append(val)
#         if len(power_history) > max_entries:
#             power_history = power_history[1:]

#         if any([power != 0 for power in power_history]):
#             draw_graph(
#                 power_history,
#                 args.mode,
#                 len(powers),
#                 args.devices,
#                 min=args.min,
#                 max=args.max,
#                 width=term_width,
#                 height=term_height,
#             )
#         else:
#             print("  -- Waiting for devices to power on...", end="\r")

#         time.sleep(args.interval)


