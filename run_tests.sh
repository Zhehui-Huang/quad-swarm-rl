#!/bin/bash

set -e
cd "$(dirname "$0")"

python -m unittest gym_art.quadrotor_multi.tests.test_body_frame_navigation
python -m gym_art.quadrotor_multi.obstacles.test.unit_test
python -m compileall -q gym_art swarm_rl

echo "Lidar obstacle traversal tests passed."
