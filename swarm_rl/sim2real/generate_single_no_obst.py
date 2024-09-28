import os
from swarm_rl.sim2real.sim2real_utils import process_layer

from swarm_rl.sim2real.code_blocks import (
    headers_network_evaluate,
    headers_evaluation,
    single_drone_eval
)


def generate_c_model(model, output_path, output_folder, testing=False):
    layer_names, bias_names, weights, biases, outputs = generate_c_weights(model, transpose=True)
    
    model_state_dict = model.state_dict()
    if 'obs_normalizer.running_mean_std.running_mean_std.obs.running_mean' in model_state_dict.keys():
        mean = model_state_dict['obs_normalizer.running_mean_std.running_mean_std.obs.running_mean']
        var = model_state_dict['obs_normalizer.running_mean_std.running_mean_std.obs.running_var']
        m_str = process_layer('mean', mean, layer_type='bias')
        v_str = process_layer('var', var, layer_type='bias')
    else:
        m_str = ""
        v_str = ""

    source = ""
    structures = ""
    methods = ""
    
    num_layers = len(layer_names)

    structure = 'static const int structure [' + str(int(num_layers)) + '][2] = {'
    for name, param in model.named_parameters():
        param = param.T
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
            structure += '{' + str(param.shape[0]) + ', ' + str(param.shape[1]) + '},'

    # complete the structure array
    # get rid of the comma after the last curly bracket
    structure = structure[:-1]
    structure += '};\n'

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
    for (int i = 0; i < structure[0][1]; i++) {{
        output_0[i] = 0;
        for (int j = 0; j < structure[0][0]; j++) {{
            output_0[i] += state_array[j] * {layer_names[0]}[j][i];
        }}
        output_0[i] += {bias_names[0]}[i];
        output_0[i] = tanhf(output_0[i]);
    }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
    for (int i = 0; i < structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n]}[i];
        output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
    }}
    '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
    for (int i = 0; i < structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n]}[i];
    }}
    '''
    for_loops.append(output_for_loop)

    # assign network outputs to control
    assignment = """
    control_n->thrust_0 = output_""" + str(n) + """[0];
    control_n->thrust_1 = output_""" + str(n) + """[1];
    control_n->thrust_2 = output_""" + str(n) + """[2];
    control_n->thrust_3 = output_""" + str(n) + """[3];
"""

    # construct the network evaluate function
    controller_eval = """\nvoid networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    for code in for_loops:
        controller_eval += code
    # assignment to control_n
    controller_eval += assignment

    # closing bracket
    controller_eval += """}"""

    # combine all the codes
    source = ""
    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    # helper funcs
    # source += linear_activation
    # source += sigmoid_activation
    # source += relu_activation
    # network eval func
    source += structure
    for output in outputs:
        source += output
    for weight in weights:
        source += weight
    for bias in biases:
        source += bias
    source += m_str
    source += v_str
    source += controller_eval

    if testing:
        source += single_drone_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


def generate_c_weights(model, transpose=False):
    """
        Generate c friendly weight strings for the c version of the single drone model
    """
    weights, biases = [], []
    layer_names, bias_names, outputs = [], [], []
    n_bias = 0
    for name, param in model.named_parameters():
        if transpose:
            param = param.T
        name = name.replace('.', '_')
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
            layer_names.append(name)
            weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
            for row in param:
                weight += '{'
                for num in row:
                    weight += str(num.item()) + ','
                # get rid of comma after the last number
                weight = weight[:-1]
                weight += '},'
            # get rid of comma after the last curly bracket
            weight = weight[:-1]
            weight += '};\n'
            weights.append(weight)

        if 'bias' in name and 'critic' not in name:
            bias_names.append(name)
            bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
            for num in param:
                bias += str(num.item()) + ','
            # get rid of comma after last number
            bias = bias[:-1]
            bias += '};\n'
            biases.append(bias)
            output = 'static float output_' + str(n_bias) + '[' + str(param.shape[0]) + '];\n'
            outputs.append(output)
            n_bias += 1

    return layer_names, bias_names, weights, biases, outputs
