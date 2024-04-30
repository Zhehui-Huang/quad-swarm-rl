# Translating torch model to c code

Currently, we support:
- [x] Single drone, w/o obstacles.
- [ ] Single drone, w/ obstacles.
- [ ] Multiple drones, w/o obstacles.
- [ ] Multiple drones, w/ obstacles.

# How to run the code
## Single drone, w/o obstacles

### Translate to c code (directly use for deployment)
We recommend use PyCharm to run the code.

--torch_model_dir=torch_models/control_50/checkpoint_p0
--output_dir=sim2real/c_models/control_50
--output_model_name=network_evaluate.c
--model_type=single
--testing=False

### Translate to c code (use for unit tests)
We use this c code to test if the c code is correct. Specifically, given same observations, is the output of c code same as the output of torch model.
The only difference between this command and the previous one is the `testing` parameter.

--torch_model_dir=torch_models/control_50/checkpoint_p0
--output_dir=sim2real/c_models/control_50
--output_model_name=network_evaluate.c
--model_type=single
--testing=True

### Unit tests
We can use the same hyperparameters that we used in Translate to c code (use for unit tests), except without the `testing` parameter.

--torch_model_dir=torch_models/control_50/checkpoint_p0
--output_dir=sim2real/c_models/control_50
--output_model_name=network_evaluate.c
--model_type=single