from ConfigParser import ConfigParser
from optparse import Values
import os
import sys

import random
import numpy.random

from ccn import data
from ccn import gpumodel


# filename: options.cfg ?
def get_options(filename, section):
    parser = ConfigParser()
    # creates a dict with a key for each section in [] in the cfg file?
    parser.read(filename)
    dirname = os.path.abspath(os.path.dirname(filename))
    options = {}
    for key, value in parser.items(section):
        value = value.replace('$HERE', dirname)
        value = value.replace('$HOME', os.path.expanduser('~'))
        options[key] = value

    if 'include' in options:
        filename_include = options['include']
        section_include = section
        if section_include not in get_sections(filename_include):
            section_include = 'DEFAULT'
        options2 = get_options(filename_include, section_include)
        options2.update(options)
        options = options2
        del options['include']

    return options


def get_sections(filename):
    parser = ConfigParser()
    parser.read(filename)
    return parser.sections()


def put_options(options_parser, opts_dict):
    options = dict((op.letter, op) for op in options_parser.options.values())
    for key, value in opts_dict.items():
        option = options.get(key)
        if option is not None:
            option.default = option.parser.parse(value)
            option.set_value(value)


def overwrite_options(opts_dict, old_op):
    for o in old_op.get_options_list():
        overwrite_value = opts_dict.get(o.prefixed_letter[2:])
        if overwrite_value is not None:
            #print 'Overwriting %s '%(o.name),
            #print `o.value` + ' with ' + `overwrite_value`
            o.value = overwrite_value
    return old_op        


def resolve(dotted):
    module, name = dotted.rsplit('.', 1)
    return getattr(__import__(module, globals(), locals(), [name], -1), name)


def handle_options(parser, section, cfg_filename):
    opts_dict = get_options(cfg_filename, section)
    data_provider = opts_dict['data-provider']
    if data_provider not in data.dp_types:
        data.DataProvider.register_data_provider(
            data_provider,
            data_provider,
            resolve(data_provider),
            )

    put_options(parser, opts_dict)
    op, load_dic = gpumodel.IGPUModel.parse_options(parser)
    op = overwrite_options(opts_dict,op)
    return op, load_dic, opts_dict


def update_attrs_from_cfg(obj, cfg, prefix):
    for key, value in cfg.items():
        if key.startswith(prefix + '.'):
            attr_name = key[len(prefix) + 1:]
            if not hasattr(obj, attr_name):
                print "%r has no attribute %r" % (obj, attr_name)
                sys.exit(1)
            attr = getattr(obj, attr_name)
            setattr(obj, attr_name, type(attr)(value))


def random_seed(seed):
    os.environ['CONVNET_RANDOM_SEED'] = str(seed)
    numpy.random.seed(seed)
    random.seed(seed)


# model_cls is instance of IGPUModel in gpumodel.py
def make_model(model_cls, section, cfg_filename=None):
    if cfg_filename is None:
        try:
            cfg_filename = sys.argv.pop(1)
        except IndexError:
            print "Provide a options configuration file as the first argument."
            sys.exit(1)
    op, load_dic, cfg = handle_options(
        parser=model_cls.get_options_parser(),
        section=section, cfg_filename=cfg_filename,
        )
    random_seed(int(cfg.get('seed', '42')))

    model = model_cls(op, load_dic)
    update_attrs_from_cfg(model, cfg, 'convnet')
    update_attrs_from_cfg(model.train_data_provider, cfg, 'dataprovider')
    update_attrs_from_cfg(model.test_data_provider, cfg, 'dataprovider') 
    return model


def run_model(model_cls, section, cfg_filename=None):
    model = make_model(model_cls, section, cfg_filename)
    model.start()

    
