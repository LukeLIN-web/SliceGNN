from microGNN.models import GAT, ScaleSAGE

models_dict = {'gat': GAT, 'sage': ScaleSAGE}


def get_model(name, params, metadata=None):
    Model = models_dict.get(name, None)
    assert Model is not None, f'Model {name} not supported!'

    if name == 'gat':
        return Model(params['inputs_channels'],
                     params['hidden_channels'],
                     params['output_channels'],
                     params['num_layers'],
                     heads=params['num_heads'])

    return Model(params['inputs_channels'], params['hidden_channels'],
                 params['output_channels'], params['num_layers'])
