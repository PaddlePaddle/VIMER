import paddle 
from paddle import optimizer as optim
from util.adamw import AdamWDL as AdamW 


def create_optimizer(args, model, filter_bias_and_bn=True, num_layers=None, skip_list=None, decay_dict=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        decay_dict = {
            param.name: not (len(param.shape) == 1 or name.endswith(".bias")
                             or name in skip_list)
            for name, param in model.named_parameters()
            if not 'teacher' in name
        } 

        parameters = [
            param for param in model.parameters()
            if 'teacher' not in param.name
        ]
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(learning_rate=args.lr, weight_decay=weight_decay)
    opt_args['parameters'] = parameters 
    if decay_dict is not None:
        opt_args['apply_decay_param_fun'] = lambda n: decay_dict[n]
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['epsilon'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['beta1'] = args.opt_betas[0]
        opt_args['beta2'] = args.opt_betas[1]
    if hasattr(args, 'layer_decay') and args.layer_decay < 1.0:
        opt_args['layerwise_decay'] = args.layer_decay
        name_dict = dict()
        for n, p in model.named_parameters():
            name_dict[p.name] = n
        opt_args['name_dict'] = name_dict 
        opt_args['n_layers'] = num_layers 
    
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('epsilon', None)
        optimizer = optim.Momentum(momentum=args.momentum, use_nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('epsilon', None)
        optimizer = optim.SGD(momentum=args.momentum, use_nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(**opt_args)
    elif opt_lower == 'adamw':
        optimizer = AdamW(**opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(**opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(alpha=0.9, momentum=args.momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer
