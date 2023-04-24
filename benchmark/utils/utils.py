from microGNN.models import GAT, SAGE, ScaleGAT, ScaleSAGE

models_dict = {'gat': GAT, 'sage': SAGE}
scale_models_dict = {'sage': ScaleSAGE, 'scalegat': ScaleGAT}


def get_model(name, params, scale=False):
    if scale:
        Model = scale_models_dict.get(name, None)
    else:
        Model = models_dict.get(name, None)
    assert Model is not None, f'Model {name} not supported!'

    if name == 'gat' or name == 'scalegat':
        return Model(params['inputs_channels'],
                     params['hidden_channels'],
                     params['output_channels'],
                     params['num_layers'],
                     heads=params['num_heads'])

    return Model(params['inputs_channels'], params['hidden_channels'],
                 params['output_channels'], params['num_layers'])
