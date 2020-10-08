import torch
import sys

from os import listdir
from models import net
from trainers import NRGANVisualizer as Visualizer

if (torch.cuda.is_available() and args.gpu_id >= 0):
       device = torch.device('cuda:%d' % args.gpu_id)
else:
       device = torch.device('cpu')

def main(saved_iter):

    args_file = open("outputs/args.txt", "r")
    data = args_file.readlines()
    args_file.close()

    args = dict()
    for line in data:
        key, val = line.split()[:2]
        args[key] = val.rstrip('\n')

    args['g_latent_dim'] = int(args['g_latent_dim'])
    args['g_image_size'] = int(args['g_image_size'])
    args['g_image_channels'] = int(args['g_image_channels'])
    args['g_channels'] = int(args['g_channels'])
    args['g_residual_factor'] = float(args['g_residual_factor'])
    args['gn_latent_dim'] = int(args['gn_latent_dim'])
    args['num_columns'] = int(args['num_columns'])
    args['implicit'] = bool(args['implicit'])
    args['rotation'] = bool(args['rotation'])
    args['channel_shuffle'] = bool(args['channel_shuffle'])
    args['color_inversion'] = bool(args['color_inversion'])

    g_params = {
            'latent_dim': args['g_latent_dim'],
            'image_size': args['g_image_size'],
            'image_channels': args['g_image_channels'],
            'channels': args['g_channels'],
            'residual_factor': args['g_residual_factor']
        }

    netG_test = net.Generator(**g_params)
    netG_test.load_state_dict(torch.load("outputs/netG_iter_" + saved_iter + ".pth"))
    netG_test.to(device)
    netG_test.eval()


    gn_params = {
            'latent_dim': args['gn_latent_dim'],
            'image_size': args['g_image_size'],
            'image_channels': args['g_image_channels'],
            'channels': args['g_channels'],
            'residual_factor': args['g_residual_factor']
        }

    netGn_test = net.Generator(**gn_params)     
    netGn_test.load_state_dict(torch.load("outputs/netGn_iter_" + saved_iter + ".pth"))
    netGn_test.to(device)   
    netGn_test.eval()      


    visualizer = Visualizer(netG_test,
                                netGn_test,
                                device,
                                args['out'],
                                args['implicit'],
                                args['prior'],
                                args['rotation'],
                                args['channel_shuffle'],
                                args['color_inversion'],
                                args['num_columns'],
                                image_range= tuple([int(i) for i in args['image_range'].replace(' ', '').strip('()').split(',')]))

    visualizer.visualize(saved_iter)
    ## visualizer.visualize(0)


if __name__ == '__main__':
    if(len(sys.argv) == 1):
        saved = None
    else:
        saved = sys.argv[1]
    main(saved)

