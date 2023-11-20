# Use this makefile to create the container for running with IPUs
default: benchmark.sif

tensorflow.sif:
	apptainer build $@ docker://docker.io/graphcore/tensorflow:2
benchmark.sif: tensorflow.sif
	apptainer build $@ def.def
