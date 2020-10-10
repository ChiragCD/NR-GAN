import argparse
import torch
from models import net
from trainers import NRGANVisualizer as Visualizer
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--gen_path', type=str, default=None)
parser.add_argument('--noiseGen_path', type=str, default=None)
parser.add_argument('--training_arg_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default='generated')
parser.add_argument('--num_images', type=int, default=2048)
visual_args = parser.parse_args()


args_file = open(visual_args.training_arg_path, "r")

data = args_file.readlines()

args_file.close()

args = dict()
for line in data:
    # print(line)
    key, val = line.split()[:2]
    args[key] = val.rstrip('\n')

# print(data)


if torch.cuda.is_available() and int(args['gpu_id']) >= 0:
   device = torch.device('cuda:%d' % int(args['gpu_id']))
else:
   device = torch.device('cpu')


g_params = {
        'latent_dim': int(args['g_latent_dim']),
        'image_size': int(args['g_image_size']),
        'image_channels': int(args['g_image_channels']),
        'channels': int(args['g_channels']),
        'residual_factor': float(args['g_residual_factor'])
    }

netG_test = net.Generator(**g_params)
netG_test.load_state_dict(torch.load(visual_args.gen_path))
netG_test.to(device)
netG_test.eval()


gn_params = {
        'latent_dim': int(args['gn_latent_dim']),
        'image_size': int(args['g_image_size']),
        'image_channels': int(args['g_image_channels']),
        'channels': int(args['g_channels']),
        'residual_factor': float(args['g_residual_factor'])
    }

netGn_test = net.Generator(**gn_params)     
netGn_test.load_state_dict(torch.load(visual_args.noiseGen_path))
netGn_test.to(device)   
netGn_test.eval()      


visualizer = Visualizer(netG_test,
                            netGn_test,
                            device,
                            visual_args.save_path,
                            bool(args['implicit']),
                            args['prior'],
                            bool(args['rotation']),
                            bool(args['channel_shuffle']),
                            bool(args['color_inversion']),
                            int(args['num_columns']),
                            image_range=(-1,1))

print("Generating {} images" .format(visual_args.num_images))
for i in tqdm(range(visual_args.num_images)):
    # visualizer = Visualizer(netG_test,
    #                         netGn_test,
    #                         device,
    #                         visual_args.save_path,
    #                         bool(args['implicit']),
    #                         args['prior'],
    #                         bool(args['rotation']),
    #                         bool(args['channel_shuffle']),
    #                         bool(args['color_inversion']),
    #                         int(args['num_columns']),
    #                         image_range=(-1,1))

    visualizer = Visualizer(netG_test,
                            netGn_test,
                            device,
                            visual_args.save_path,
                            bool(args['implicit']),
                            args['prior'],
                            bool(args['rotation']),
                            bool(args['channel_shuffle']),
                            bool(args['color_inversion']),
                            num_columns = 1,
                            image_range=(-1,1))


    visualizer.visualize(i)



