import time

import pandas as pd
from multiprocessing import Process, Queue, Event
import sys

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
    
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()
        
        interval = 100 #ms
        self.smip = Process(target=self._power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self


    def pow_to_float(self,pow):
        # Power is reported in the format xxx.xxW, so remove the last character.
        # We also handle the case when the power reports as N/A.
        try:
            return float(pow[:-1])
        except ValueError:
            return 0
    
    def _power_loop(self,queue, event, interval):
        import gcipuinfo
        

        ipu_info = gcipuinfo.gcipuinfo()
        num_devices = len(ipu_info.getDevices())
        
        power_value_dict = {
            idx : [] for idx in range(num_devices)
        }
        power_value_dict['timestamps'] = []
       
        last_timestamp = time.time()

        while not event.is_set():
            #for idx in range(num_devices):
            gcipuinfo.IpuPower
            device_powers=ipu_info.getNamedAttributeForAll(gcipuinfo.IpuPower)
            device_powers = [self.pow_to_float(pow) for pow in device_powers if pow != "N/A"]
            for idx in range(num_devices):
                power_value_dict[idx].append(device_powers[idx])
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


    


