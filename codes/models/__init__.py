import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model =='AttenGAN':
        from .AttenGAN_model import AttenGANModel as M
    elif model =='ShareGAN':
        from .ShareGAN_model import ShareGANModel as M
    elif model == 'ShareGAN_RCAN':
        from .ShareSR_model import ShareSRModel as M
    elif model == 'DualSR':
        from .DualSR_model import DualSRModel as M
    elif model == 'SASRGAN':
        from .SASRGAN_model import SASRGANModel as M
    elif model == 'DualSR_pretrain':
        from .DualSR_pretrain import DualSR_pretrain as M
    elif model == 'DualSR_pretrain_high':
        from .DualSR_pretrain_high import DualSR_pretrain_high as M
    elif model == 'DualSR_pretrain_mask':
        from .DualSR_pretrain_mask import DualSR_pretrain_mask as M
    elif model == 'DualSR_high':
        from .DualSR_model_high import DualSRModel_high as M
    elif model == 'DualSR_mask':
        from .DualSR_model_mask import DualSRModel_mask as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
