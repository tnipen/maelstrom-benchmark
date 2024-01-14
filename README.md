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


To run JUBE, first edit jube/jube.yaml to set the settings for the benchmark. Then run JUBE like this:
```
cd jube/
jube run jube.yaml -t jwb
```
Adapt the tag for the respective hardware as listed above.

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
