import os

from swarm_rl.sim2real.code_blocks import headers_network_evaluate, headers_evaluation
from swarm_rl.sim2real.code_blocks import headers_single_obst_1, headers_single_obst_2, \
    headers_single_obst_3, headers_single_obst_4, multi_drone_attn_eval, normalization_functions
from swarm_rl.sim2real.code_blocks import headers_multi_obst_deepset_1, headers_multi_obst_deepset_2, \
    headers_multi_obst_deepset_3
from swarm_rl.sim2real.sim2real_utils import process_layer


def generate_c_model_deepset_obst(model, output_path, output_folder, testing=False):
    info = generate_c_weights_deepset_obst(model, transpose=True)
    model_state_dict = model.state_dict()

    if 'obs_normalizer.running_mean_std.running_mean_std.obs.running_mean' in model_state_dict.keys():
        mean = model_state_dict['obs_normalizer.running_mean_std.running_mean_std.obs.running_mean']
        var = model_state_dict['obs_normalizer.running_mean_std.running_mean_std.obs.running_var']
        m_str = process_layer('mean', mean, layer_type='bias')
        v_str = process_layer('var', var, layer_type='bias')
        norm_method = normalization_functions
    else:
        m_str = ""
        v_str = ""
        norm_method = ""

    source = ""
    structures = ""
    methods = ""

    # setup all the encoders
    for enc_name, data in info['encoders'].items():
        # data contains [weight_names, bias_names, weights, biases]
        structure = f'static const int {enc_name}_structure [' + str(int(len(data[0]))) + '][2] = {'

        weight_names, bias_names = data[0], data[1]
        for w_name, b_name in zip(weight_names, bias_names):
            w = model_state_dict[w_name].T
            structure += '{' + str(w.shape[0]) + ', ' + str(w.shape[1]) + '},'
            
        if 'self' in enc_name:
            self_enc_size = int(structure.split(",")[4].split("{")[-1])
        if 'obst' in enc_name:
            obst_enc_size = int(structure.split(",")[-2][:-1])
            obst_dim = int(structure.split(",")[0].split("{")[-1])
        if 'nbr' in enc_name:
            nbr_enc_size = int(structure.split(",")[-2][:-1])
            nbr_dim = int(structure.split(",")[0].split("{")[-1])

        # complete the structure array
        # get rid of the comma after the last curly bracket
        structure = structure[:-1]
        structure += '};\n'
        structures += structure

        if 'self' in enc_name:
            method = self_encoder_c_str(enc_name, weight_names, bias_names)
        elif 'nbr' in enc_name:
            method = neighbor_encoder_deepset_c_string(enc_name, weight_names, bias_names, testing)
        elif 'obst' in enc_name:
            method = obstacle_encoder_c_str(enc_name, weight_names, bias_names)
            
            
        methods += method

    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    source += headers_multi_obst_deepset_1 + str(nbr_dim)
    source += headers_single_obst_1 + str(obst_dim) + headers_single_obst_2 + str(obst_enc_size) + headers_single_obst_3 + str(self_enc_size) + headers_single_obst_4
    source += headers_multi_obst_deepset_2 + str(nbr_enc_size) + headers_multi_obst_deepset_3

    # helper funcs
    # source += linear_activation
    # source += sigmoid_activation
    # source += relu_activation

    # network eval func
    source += structures
    outputs = info['outputs']
    for output in outputs:
        source += output

    encoders = info['encoders']

    for key, vals in encoders.items():
        weights, biases = vals[-2], vals[-1]
        for w in weights:
            source += w
        for b in biases:
            source += b

    source += m_str
    source += v_str
    
    source += norm_method

    source += methods

    if testing:
        source += multi_drone_attn_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


def generate_c_weights_deepset_obst(model, transpose=False):
    """
            Generate c friendly weight strings for the c version of the attention model
            order is: self-encoder, neighbor-encoder, obst-encoder, attention, then final combined output layers
    """
    self_weights, self_biases, self_layer_names, self_bias_names = [], [], [], []
    neighbor_weights, neighbor_biases, nbr_layer_names, nbr_bias_names = [], [], [], []
    obst_weights, obst_biases, obst_layer_names, obst_bias_names = [], [], [], []
    out_weights, out_biases, out_layer_names, out_bias_names = [], [], [], [],
    outputs = []
    n_self, n_nbr, n_obst = 0, 0, 0
    for name, param in model.named_parameters():
        # get the self encoder weights
        if transpose:
            param = param.T
        c_name = name.replace('.', '_')
        if 'weight' in c_name and 'critic' not in c_name and 'layer_norm' not in c_name:
            weight = process_layer(c_name, param, layer_type='weight')
            if 'self_embed' in c_name:
                self_layer_names.append(name)
                self_weights.append(weight)
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
            elif 'neighbor_encoder' in c_name:
                nbr_layer_names.append(name)
                neighbor_weights.append(weight)
                outputs.append(
                    'static float nbr_output_' + str(n_nbr) + '[NEIGHBORS]' + '[' + str(param.shape[1]) + '];\n')
                n_nbr += 1
            elif 'obstacle' in c_name:
                obst_layer_names.append(name)
                obst_weights.append(weight)
                outputs.append('static float obst_output_' + str(n_obst) + '[' + str(param.shape[1]) + '];\n')
                n_obst += 1
            else:
                # output layer
                out_layer_names.append(name)
                out_weights.append(weight)
                # these will be considered part of the self encoder
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
        if ('bias' in c_name or 'layer_norm' in c_name) and 'critic' not in c_name:
            bias = process_layer(c_name, param, layer_type='bias')
            if 'self_embed' in c_name:
                self_bias_names.append(name)
                self_biases.append(bias)
            elif 'neighbor' in c_name:
                nbr_bias_names.append(name)
                neighbor_biases.append(bias)
            elif 'obstacle' in c_name:
                obst_bias_names.append(name)
                obst_biases.append(bias)
            else:
                # output layer
                out_bias_names.append(name)
                out_biases.append(bias)

    self_layer_names += out_layer_names
    self_bias_names += out_bias_names
    self_weights += out_weights
    self_biases += out_biases
    info = {
        'encoders': {
            'self': [self_layer_names, self_bias_names, self_weights, self_biases],
            'nbr': [nbr_layer_names, nbr_bias_names, neighbor_weights, neighbor_biases],
            'obst': [obst_layer_names, obst_bias_names, obst_weights, obst_biases],
        },
        'out': [out_layer_names, out_bias_names, out_weights, out_biases],
        'outputs': outputs
    }

    return info

def self_encoder_c_str(prefix, weight_names, bias_names):
    method = """\nvoid networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
    for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
        output_0[i] = 0;
        for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
            output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
        }}
        output_0[i] += {bias_names[0].replace('.', '_')}[i];
        output_0[i] = tanhf(output_0[i]);
    }}
'''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 2):
        for_loop = f'''
    for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
        output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
    }}
'''
    for_loops.append(for_loop)
    
    # Concat self embedding, neighbor embedding and obstacle embedding
    for_loop = f'''
    // Concat self_embed, neighbor_embed and obst_embed
    for (int i = 0; i < self_structure[1][1]; i++) {{
        output_embeds[i] = output_1[i]; // Self
        output_embeds[i + self_structure[1][1]] = obstacle_embeds[i]; // Obstacles
        output_embeds[i + self_structure[1][1] + nbr_structure[1][1]] = neighbor_embeds[i]; // Neighbors
    }}
    '''
    for_loops.append(for_loop)

    # forward-prop of feedforward layer
    output_for_loop = f'''
    // Feedforward layer
    for (int i = 0; i < self_structure[2][1]; i++) {{
        output_{num_layers - 2}[i] = 0;
        for (int j = 0; j < self_structure[2][0] ; j++) {{
            output_{num_layers - 2}[i] += output_embeds[j] * actor_encoder_feed_forward_0_weight[j][i];
        }}
        output_{num_layers - 2}[i] += actor_encoder_feed_forward_0_bias[i];
        output_{num_layers - 2}[i] = tanhf(output_2[i]);
    }}
'''
    for_loops.append(output_for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
    for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
    }}
'''
    for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
    control_n->thrust_0 = output_""" + str(n) + """[0];
    control_n->thrust_1 = output_""" + str(n) + """[1];
    control_n->thrust_2 = output_""" + str(n) + """[2];
    control_n->thrust_3 = output_""" + str(n) + """[3];
"""
        method += assignment
    # closing bracket
    method += """}\n\n"""
    return method


def neighbor_encoder_deepset_c_string(prefix, weight_names, bias_names, testing):
    if testing:
        method = """void neighborEmbedder(const float neighbor_inputs[NEIGHBORS*NBR_OBS_DIM]) {"""
    else:
        method = """void neighborEmbedder(volatile float neighbor_inputs[NEIGHBORS*NBR_OBS_DIM]) {"""
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
    for (int n = 0; n < NEIGHBORS; n++) {{            
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            {prefix}_output_0[n][i] = 0; 
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                {prefix}_output_0[n][i] += neighbor_inputs[n*NBR_OBS_DIM + j] * actor_encoder_neighbor_encoder_embedding_mlp_0_weight[j][i]; 
            }}
            {prefix}_output_0[n][i] += actor_encoder_neighbor_encoder_embedding_mlp_0_bias[i];
            {prefix}_output_0[n][i] = tanhf({prefix}_output_0[n][i]);
        }}
    }}
'''
    for_loops.append(input_for_loop)

    # hidden layers
    for n in range(1, num_layers):
        for_loop = f'''
    for (int n = 0; n < NEIGHBORS; n++) {{
        for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
            {prefix}_output_{str(n)}[n][i] = 0;
            for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                {prefix}_output_{str(n)}[n][i] += {prefix}_output_{str(n - 1)}[n][j] * {weight_names[n].replace('.', '_')}[j][i];
            }}
            {prefix}_output_{str(n)}[n][i] += {bias_names[n].replace('.', '_')}[i];
            {prefix}_output_{str(n)}[n][i] = tanhf({prefix}_output_{str(n)}[n][i]);
        }}
    }}
'''
        for_loops.append(for_loop)

    # Average the neighbor embeddings
    for_loop = f'''
    // Average over number of neighbors
    for (int i = 0; i < NBR_OBS_DIM; i++) {{
        neighbor_embeds[i] = 0;
        for (int n = 0; n < NEIGHBORS; n++) {{
            neighbor_embeds[i] += {prefix}_output_{str(num_layers - 1)}[n][i];
        }}
        neighbor_embeds[i] /= NEIGHBORS;
    }}
'''
    for_loops.append(for_loop)

    for code in for_loops:
        method += code
    # method closing bracket
    method += """}\n"""
    return method


def obstacle_encoder_c_str(prefix, weight_names, bias_names):
    method = f"""void obstacleEmbedder(volatile float obstacle_inputs[OBST_DIM]) {{
            //reset embeddings accumulator to zero
            memset(obstacle_embeds, 0, sizeof(obstacle_embeds));

        """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
                {prefix}_output_0[i] = 0;
                for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                    {prefix}_output_0[i] += obstacle_inputs[j] * {weight_names[0].replace('.', '_')}[j][i];
                }}
                {prefix}_output_0[i] += {bias_names[0].replace('.', '_')}[i];
                {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
            }}
        '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers):
        for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                {prefix}_output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    {prefix}_output_{str(n)}[i] += {prefix}_output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                {prefix}_output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                obstacle_embeds[i] = tanhf({prefix}_output_{str(n)}[i]);
            }}
            '''
        for_loops.append(for_loop)

    for code in for_loops:
        method += code
    # closing bracket
    method += """}\n\n"""
    return method
