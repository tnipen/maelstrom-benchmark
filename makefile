# Use this makefile to create the container for running with IPUs/MI250x/H100
CACHEDIR=/p/project/deepacf/maelstrom/apptainer/cache/
TMPDIR=/p/project/deepacf/maelstrom/apptainer/tmpdir/

ipu.sif:
	APPTAINER_CACHEDIR=$(CACHEDIR) apptainer pull --tmpdir $(TMPDIR) docker://docker.io/graphcore/tensorflow:2
	apptainer build $@ ipu.def
mi250x.sif:
	APPTAINER_CACHEDIR=$(CACHEDIR) apptainer pull --tmpdir $(TMPDIR) docker://rocm/tensorflow:rocm5.7-tf2.13-dev
	apptainer build $@ mi250x.def
h100.sif:
	APPTAINER_CACHEDIR=$(CACHEDIR) apptainer pull --tmpdir $(TMPDIR) docker://nvcr.io/nvidia/tensorflow:23.04-tf2-py3
	apptainer build $@ h100.def
