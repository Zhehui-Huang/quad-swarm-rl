#!/usr/bin/env python

headers_controller_nn = """
#include "math3d.h"
#include "stabilizer_types.h"
#include <math.h>
#include "controller_nn.h"

"""

headers_network_evaluate = """#include "network_evaluate_tof.h"

"""

headers_multi_agent_mean_embed = """#include "network_evaluate_tof.h"

"""

headers_single_obst_1 = """
#define EPS 0.000001 // 1e-6
#define OBST_DIM """

headers_single_obst_2 = """

static float obstacle_embeds["""

headers_single_obst_3 = """];
static float output_embeds["""

headers_single_obst_4 = """];

float base;
float exponent;
"""

headers_multi_agent_attention = """
// attention stuff
#define D_MODEL 12
#define NUM_TOKENS 2
#define EPS 0.000001 // 1e-6

static float tokens[NUM_TOKENS][D_MODEL];
static float q_outputs[NUM_TOKENS][D_MODEL];
static float k_outputs[NUM_TOKENS][D_MODEL];
static float v_outputs[NUM_TOKENS][D_MODEL];
static float residual[NUM_TOKENS][D_MODEL];

static float attn_weights[NUM_TOKENS][NUM_TOKENS];
static float softmax_sums[NUM_TOKENS];
static float query_output[NUM_TOKENS][D_MODEL];
static float query_embeds[NUM_TOKENS][D_MODEL];

static float last_layer_means[NUM_TOKENS];
static float last_layer_variances[NUM_TOKENS];
static float attn_embeds[NUM_TOKENS][D_MODEL];

static float output_embeds[3 * D_MODEL];
static float obstacle_embeds[12];
static float neighbor_embeds[12];

float base;
float exponent;
"""

headers_multi_deepset_evaluation = """#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring> 

typedef struct control_t_n {
	float thrust_0;
	float thrust_1;
	float thrust_2;
	float thrust_3;
} control_t_n;

void networkEvaluate(control_t_n* control_n, const float* state_array);

"""

headers_evaluation = """#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring> 

typedef struct control_t_n {
	float thrust_0;
	float thrust_1;
	float thrust_2;
	float thrust_3;
} control_t_n;

void networkEvaluate(control_t_n* control_n, const float* state_array);

// static const int NEIGHBORS = 2;
// static const int NBR_OBS_DIM = 6; 
 
// static const int NUM_OBSTACLES = 2; 
// static const int OBST_DIM = 9;

"""


constants = """

#define MAX_THRUST 0.1597
// PWM to thrust coefficients
#define A 2.130295e-11
#define B 1.032633e-6
#define C 5.484560e-4

"""

controller_init_function = """

void controllerNNInit(void) {}

"""

controller_test_function = """

bool controllerNNTest(void) {
	return true;
}

"""

linear_activation = """

float linear(float num) {
	return num;
}

"""

sigmoid_activation = """

float sigmoid(float num) {
	return 1 / (1 + exp(-num));
}

"""

relu_activation = """

float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

"""

scaling = """
// range of action -1 ... 1, need to scale to range 0 .. 1
float scale(float v) {
	return 0.5f * (v + 1);
}

"""

clipping = """

float clip(float v, float min, float max) {
	if (v < min) return min;
	if (v > max) return max;
	return v;
}

"""

attention_body = """
void singleHeadAttention() {
        uint8_t i, j, k;
        // fill the tokens matrix with obstacle and neighbor embeddings
        for (i = 0; i < D_MODEL; i++) {
            tokens[0][i] = neighbor_embeds[i];
            tokens[1][i] = obstacle_embeds[i];
        }
        
        // save the residual
        for (i = 0; i < NUM_TOKENS; i++) {
            for (j = 0; j < D_MODEL; j++) {
                residual[i][j] = tokens[i][j];
            }
        }
        
        // compute Q, K, V projection 
        for (i = 0; i < NUM_TOKENS; i++) {
            for (j = 0; j < D_MODEL; j++) {
                q_outputs[i][j] = 0;
                k_outputs[i][j] = 0;
                v_outputs[i][j] = 0;
                for (k = 0; k < D_MODEL; k++) {
                    q_outputs[i][j] += tokens[i][k] * actor_encoder_attention_layer_w_qs_weight[k][j]; 
                    k_outputs[i][j] += tokens[i][k] * actor_encoder_attention_layer_w_ks_weight[k][j];
                    v_outputs[i][j] += tokens[i][k] * actor_encoder_attention_layer_w_vs_weight[k][j];
                }
            }
        }
        
        // compute attention weights and parts of softmax
        for (i = 0; i < NUM_TOKENS; i++) {
            softmax_sums[i] = 0;
            for (int j = 0; j < NUM_TOKENS; j++) {
                attn_weights[i][j] = 0;
                for (k = 0; k < D_MODEL; k++) {
                    attn_weights[i][j] += (q_outputs[i][k] / sqrt((float)D_MODEL)) * k_outputs[j][k];
                }
                softmax_sums[i] += exp(attn_weights[i][j]);
            }
        }
        
        // softmax
        for (i = 0; i < NUM_TOKENS; i++) {
            for (j = 0; j < NUM_TOKENS; j++) {
                attn_weights[i][j] = exp(attn_weights[i][j]) / softmax_sums[i];
            }
        }

        // compute query 
        for (i = 0; i < NUM_TOKENS; i++) {
            for (j = 0; j < D_MODEL; j++) {
                query_output[i][j] = 0;
                for (k = 0; k < NUM_TOKENS; k++) {
                    query_output[i][j] += attn_weights[i][k] * v_outputs[k][j];
                }
            }
        }
        
        // compute the q embeddings 
        for (i = 0; i < NUM_TOKENS; i++) {
            for (j = 0; j < D_MODEL; j++) {
                query_embeds[i][j] = 0;
                for (k = 0; k < D_MODEL; k++) {
                    query_embeds[i][j] += query_output[i][k] * actor_encoder_attention_layer_fc_weight[k][j];
                }
                // add the residual
                query_embeds[i][j] += residual[i][j];
            }
        }
        
        // calculate per-token mean
        for (i = 0; i < NUM_TOKENS; i++){
            last_layer_means[i] = 0;
            for (j = 0; j < D_MODEL; j++) {
                last_layer_means[i] +=  query_embeds[i][j];
            }
            last_layer_means[i] = last_layer_means[i] / (float)D_MODEL;
        }
        
        // calculate per-token variance
        for (i = 0; i < NUM_TOKENS; i++) {
            last_layer_variances[i] = 0;
            for (j = 0; j < D_MODEL; j++) {
                base = query_embeds[i][j] - last_layer_means[i];
                exponent = pow(base, 2);
                last_layer_variances[i] += exponent;
            }
            last_layer_variances[i] = last_layer_variances[i] / (float)(D_MODEL);
        }
        
        // perform layer norm (2x16) x (16)
        for (i = 0; i < NUM_TOKENS; i++) {
            for (j = 0; j < D_MODEL; j++) {
                attn_embeds[i][j] = (query_embeds[i][j] - last_layer_means[i]) / (sqrt(last_layer_variances[i] + EPS)) * actor_encoder_attention_layer_layer_norm_weight[j]; 
                attn_embeds[i][j] += actor_encoder_attention_layer_layer_norm_bias[j]; 
            }
        }
}
"""


controller_entry = """
static control_t_n control_n;
static struct mat33 rot;
static float state_array[18];

void controllerNN(control_t *control, 
				  setpoint_t *setpoint, 
				  const sensorData_t *sensors, 
				  const state_t *state, 
				  const uint32_t tick)
{
	control->enableDirectThrust = true;
	if (!RATE_DO_EXECUTE(RATE_250_HZ, tick)) {
		return;
	}

	// Orientation
	struct quat q = mkquat(state->attitudeQuaternion.x, 
						   state->attitudeQuaternion.y, 
						   state->attitudeQuaternion.z, 
						   state->attitudeQuaternion.w);
	rot = quat2rotmat(q);

	// angular velocity
	float omega_roll = radians(sensors->gyro.x);
	float omega_pitch = radians(sensors->gyro.y);
	float omega_yaw = radians(sensors->gyro.z);

	// the state vector
	state_array[0] = state->position.x - setpoint->position.x;
	state_array[1] = state->position.y - setpoint->position.y;
	state_array[2] = state->position.z - setpoint->position.z;
	state_array[3] = state->velocity.x;
	state_array[4] = state->velocity.y;
	state_array[5] = state->velocity.z;
	state_array[6] = rot.m[0][0];
	state_array[7] = rot.m[0][1];
	state_array[8] = rot.m[0][2];
	state_array[9] = rot.m[1][0];
	state_array[10] = rot.m[1][1];
	state_array[11] = rot.m[1][2];
	state_array[12] = rot.m[2][0];
	state_array[13] = rot.m[2][1];
	state_array[14] = rot.m[2][2];
	state_array[15] = omega_roll;
	state_array[16] = omega_pitch;
	state_array[17] = omega_yaw;
	state_array[18] = state->position.z;


	// run the neural neural network
	networkEvaluate(&control_n, state_array);

	// convert thrusts to directly to PWM
	// need to hack the firmware (stablizer.c and power_distribution_stock.c)
	int PWM_0, PWM_1, PWM_2, PWM_3; 
	thrusts2PWM(&control_n, &PWM_0, &PWM_1, &PWM_2, &PWM_3);

	control->motorRatios[0] = PWM_0;
	control->motorRatios[1] = PWM_1;
	control->motorRatios[2] = PWM_2;
	control->motorRatios[3] = PWM_3;
}


/*
 * Crazyflie rotors positions (as opposed to what the neural network assumes): 
 *							x 
 *							|
 *						3	|	0
 *				 (thrust 0)	|	(thrust 1)
 *				  y <---------------
 *					 		|
 *					 	2	|	1
 *			     (thrust 3) |	(thrust 2)
 *	(thrust projection must align with each motor, e.g thrust 1 is on rotor 0)
 */
void thrusts2PWM(struct control_t_n *control_n, 
	int *PWM_0, int *PWM_1, int *PWM_2, int *PWM_3){

	/*
	// scaling and cliping
	control_n->thrust_0 = MAX_THRUST * clip(scale(control_n->thrust_0), 0.0, 1.0);
	control_n->thrust_1 = MAX_THRUST * clip(scale(control_n->thrust_1), 0.0, 1.0);
	control_n->thrust_2 = MAX_THRUST * clip(scale(control_n->thrust_2), 0.0, 1.0);
	control_n->thrust_3 = MAX_THRUST * clip(scale(control_n->thrust_3), 0.0, 1.0);

	// motor 0
	*PWM_0 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_1))) / (2 * A);
	// motor 1
	*PWM_1 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_2))) / (2 * A);
	// motor 2
	*PWM_2 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_3))) / (2 * A);
	// motor 3 
	*PWM_3 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_0))) / (2 * A);
	*/

	// scaling and cliping
	control_n->thrust_0 = UINT16_MAX * clip(scale(control_n->thrust_0), 0.0, 1.0);
	control_n->thrust_1 = UINT16_MAX * clip(scale(control_n->thrust_1), 0.0, 1.0);
	control_n->thrust_2 = UINT16_MAX * clip(scale(control_n->thrust_2), 0.0, 1.0);
	control_n->thrust_3 = UINT16_MAX * clip(scale(control_n->thrust_3), 0.0, 1.0);

	// motor 0
	*PWM_0 = control_n->thrust_0;
	// motor 1
	*PWM_1 = control_n->thrust_1;
	// motor 2
	*PWM_2 = control_n->thrust_2;
	// motor 3 
	*PWM_3 = control_n->thrust_3;
}

"""

log_group = """

LOG_GROUP_START(ctrlNN)
LOG_ADD(LOG_FLOAT, thrust0, &output_2[0])
LOG_ADD(LOG_FLOAT, thrust1, &output_2[1])
LOG_ADD(LOG_FLOAT, thrust2, &output_2[2])
LOG_ADD(LOG_FLOAT, thrust3, &output_2[3])
LOG_GROUP_STOP(ctrlNN)

"""

single_drone_eval = """

int main(const float *indatav, size_t size, float *outdatav)
{
    size_t i;
    control_t_n motorThrusts;
    networkEvaluate(&motorThrusts, indatav);

    outdatav[0] = motorThrusts.thrust_0;
    outdatav[1] = motorThrusts.thrust_1;
    outdatav[2] = motorThrusts.thrust_2;
    outdatav[3] = motorThrusts.thrust_3;
    return EXIT_SUCCESS;
}

"""

single_drone_obst_eval = """

int main(const float *indatav, size_t size, float *outdatav)
{
    size_t i;
    control_t_n motorThrusts;
    
    
    networkEvaluate(&motorThrusts, indatav);

    outdatav[0] = motorThrusts.thrust_0;
    outdatav[1] = motorThrusts.thrust_1;
    outdatav[2] = motorThrusts.thrust_2;
    outdatav[3] = motorThrusts.thrust_3;
    return EXIT_SUCCESS;
}

"""

multi_drone_deepset_eval = """
int main(const float *self_indatav, const float *nbr_indatav, float *outdatav){
    size_t i;
    control_t_n motorThrusts;

    neighborEmbedder(nbr_indatav);
    
    networkEvaluate(&motorThrusts, self_indatav);

    outdatav[0] = motorThrusts.thrust_0;
    outdatav[1] = motorThrusts.thrust_1;
    outdatav[2] = motorThrusts.thrust_2;
    outdatav[3] = motorThrusts.thrust_3;
    return EXIT_SUCCESS;
}
"""

single_drone_obst_eval_1 = """
int main(const float *self_indatav, const float *obst_indatav, float *obst_outdata, float *outdatav){
    size_t i;
    control_t_n motorThrusts;

    obstacleEmbedder(obst_indatav);

    networkEvaluate(&motorThrusts, self_indatav);
    
    for (int i = 0; i < """
single_drone_obst_eval_2 = """; i++) {
        obst_outdata[i] = obstacle_embeds[i];
    }

    outdatav[0] = motorThrusts.thrust_0;
    outdatav[1] = motorThrusts.thrust_1;
    outdatav[2] = motorThrusts.thrust_2;
    outdatav[3] = motorThrusts.thrust_3;
    return EXIT_SUCCESS;
}
"""

multi_drone_attn_eval = """

int main(const float *self_indatav, const float *nbr_indatav, float *obst_indatav, float *nbr_outdata, float *obst_outdata, float *token1_out, float *token2_out, float *outdatav)
{
    size_t i;
    control_t_n motorThrusts;

    obstacleEmbedder(obst_indatav); 
    neighborEmbedder(nbr_indatav);
    singleHeadAttention();
    
    for (int i = 0; i < D_MODEL; i++) {
        obst_outdata[i] = obst_output_0[i];
        nbr_outdata[i] = nbr_output_0[i];  
        token1_out[i] = attn_embeds[0][i];
        token2_out[i] = attn_embeds[1][i]; 
    }

    networkEvaluate(&motorThrusts, self_indatav);

    outdatav[0] = motorThrusts.thrust_0;
    outdatav[1] = motorThrusts.thrust_1;
    outdatav[2] = motorThrusts.thrust_2;
    outdatav[3] = motorThrusts.thrust_3;
    
    return EXIT_SUCCESS; 
}

"""