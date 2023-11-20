# MAELSTROM hardware benchmark

This repo contains a benchmark script to run MAELSTROM applications on several different hardwares at JSC. Currently, CPU, GPU, and IPU (Graphcore) hardware are supported.

### Installation
Run `make` to build the container for running on IPUs on JURECA

### Running the benchmarks with JUBE
The JUBE file contains 4 tags:
- ipu. This must be scheduled from a JURECA login node.
- jwb (using A100 GPUs). This must be scheduled from a JUWELS-booster login node.
- amd (using AMD CPUs). This must be scheduled from a JUWELS-booster login node.
- jwc (using V100 GPUs). This must be scheduled from a JUWELS-cluster login node.
- intel (using Intel CPUs). This must be scheduled from a JUWELS-cluster login node.

To run JUBE, first edit jube/jube.yaml to set the settings for the benchmark. Then run JUBE like this:
```
cd jube/
jube run jube.yaml -t jwb
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
