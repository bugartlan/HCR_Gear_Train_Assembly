#!/usr/bin/env bash

cd "$(dirname "$0")/local_ray"
./ray_interface.sh "${@:1}"
