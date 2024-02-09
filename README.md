# MAELSTROM hardware benchmark

This repo contains a benchmark script to run MAELSTROM applications on several different hardwares at JSC. Currently, CPU, GPU, and IPU (Graphcore) hardware are supported.

### Installation
Run `make` to build the container for running on the IPU, H100 or MI250x.
```
make ipu.sif    # create container for IPU
make h100.sif   # create container for H100
make mi250x     # create container for MI250x
```

### Running the benchmarks with JUBE
JUBE is made available via modules on JSC's and E4's HPC-cluster. <br>
On JSC clusters, just run:
```
ml JUBE/2.6.1
```
On E4-clusters run:
```
module use /opt/share/users/testusr/modulefiles
module load jube
```
The JUBE file contains 9 tags:
- ipu. This must be scheduled from a JURECA login node.
- jwb (using A100 GPUs). This must be scheduled from a JUWELS-booster login node.
- amd (using AMD CPUs). This must be scheduled from a JUWELS-booster login node.
- jwc (using V100 GPUs). This must be scheduled from a JUWELS-cluster login node.
- intel (using Intel CPUs). This must be scheduled from a JUWELS-cluster login node.
- h100. This must be scheduled from a JURECA login node.
- mi250x. This must be scheduled from a JURECA login node.
- e4a2. This must be scheduled from the E4 machines 
- e4gh200. This must be scheduled from the E4 machines.

Do the following to run JUBE for AP5 (here on Juwels Booster's A100-nodes):
```
cd jube/
jube run jube_ap5.yaml -t jwb
```

Adapt the tag for the respective hardware as listed above.

### Experimenting on the H100-node
Since the last update of CUDA-driver around the 20th January, s strong performance degradation has been noted on the H100-nodes.
While this issue is still under investigation, two singularity container versions are made available:

```
make h100.sif
make h100_tf23.04.sif
```
where the former (latter) is based on Nvidia's provided Tensorflow image version 23.10 (23.04).
The 23.10-version is deployed by default when running on the H100-node. To choose the 23.04-version, a corresponding tag has to be added to the jube-command:

```
jube run jube_ap5.yaml -t h100 tf23.04
```
So far, better performance has been noted with ```keras.optimizers.legacy.Adam``` instead of the standard optimizer. However, this optimizer is not used by default and requires manual modification of the Python-script ```../applications.py```. 
In particular, the l.332 (l.331) must be (un-)commented.
```
331            opt = keras.optimizers.legacy.Adam      # so far, better performance for H100-experiments
332            #opt = keras.optimizers.Adam            # default
```

### Running the benchmark manually

Run the benchmark script as follows:

```
python3 benchmark.py [options]
```

See available options by running:

```
python3 benchmark.py --help
```

### Adding a new application

1) Add a new subclass of Application in `applications.py`.
2) Add a line in the `get` function to connect your application.
3) Change the "app" parameter in the JUBE file to match your application.
