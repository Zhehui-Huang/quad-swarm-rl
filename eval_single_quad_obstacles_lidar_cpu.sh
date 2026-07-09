#!/bin/bash

set -e
cd "$(dirname "$0")"

exec bash eval_single_quad_obstacles_lidar.sh "$@"
