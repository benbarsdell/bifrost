# Bifrost

A stream processing framework for high-throughput applications.

Bifrost aims to be easy to use, easy to debug, performant enough to
compete with best-in-class, and flexible enough to implement even
advanced pipeline designs.

## Feature overview

 * Designed for sustained high-throughput stream processing (offline batch processing also possible).
 * High-level interface: JSON pipeline descriptions and simple C++ plugins.
 * Strong decoupling of processing tasks: unconstrained buffering and data shape/type interpretation.
 * Built-in support for both system (CPU) and CUDA (GPU) memory spaces and computation.
 * Built-in support for broadcasting of real-time monitoring data, plus tools for subscribing to and visualizing these data.
 * Library of plugins for common processing operations (e.g., packetize/depacketize, FFT, BLAS; potentially many more).

## Design overview

Bifrost Pipelines, described in JSON format, consist of a graph of
multiple-input multiple-output Tasks connected via flexible
RingBuffers. When launched, each Task runs asynchronously and
processes data from its inputs (if any) to its outputs (if any).

## Installation

    $ make

will build libbifrost.so, the bifrost_launch application and the standard plugins.

## Contributors

 * Ben Barsdell (gmail benbarsdell)
 * Daniel Price (thetelegraphic.com dan)
