import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits)
    elif opt['data.dataset'] == 'vctk':
        ds = protonets.data.vctk.load(opt, splits)
    elif opt['data.dataset'] == 'vctk_few_shot':
        ds = protonets.data.vctk.load(opt, splits, for_few_shot=True)
    elif opt['data.dataset'] == 'voxceleb':
        ds = protonets.data.VoxCeleb.load(opt, splits)
    elif opt['data.dataset'] == 'VoxCeleb2':
        ds = protonets.data.VoxCeleb2.load(opt, splits)
    elif opt['data.dataset'] == 'LibriSpeech':
        ds = protonets.data.LibriSpeech.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
