# Parameters for direct parameter passing

import argparse

parser = argparse.ArgumentParser(description="Go CNSDiff")
parser.add_argument('--n_candidates', type=int, default=1)
parser.add_argument('--gpu', type=int, default=1,
                    help='GPU device ID to use (default: auto-detect)')
parser.add_argument('--n_hid', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')

# params for the denoiser - learn an MLP for denoising
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=int, default=64, help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=64, help='timestep embedding size')

# Parameters required for diffusion model generation
parser.add_argument('--steps', type=int, default=20, help='diffusion steps')  #20
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating') # original value
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True,
                    help='assign different weight to different timestep or not')

# LightGCN training related parameters
parser.add_argument('--initial_alpha', type=int, default=1, help='Initial alpha parameter for Beta distribution')
parser.add_argument('--initial_beta', type=int, default=9, help='Initial beta parameter for Beta distribution')
parser.add_argument('--final_alpha', type=int, default=9, help='Final alpha parameter for Beta distribution')
parser.add_argument('--final_beta', type=int, default=1, help='Final beta parameter for Beta distribution')
parser.add_argument('--cl_weight', type=float, default=0.3, help='Contrastive learning loss weight')
parser.add_argument('--min_mix_ratio', type=float, default=0, help='Minimum mixing ratio for negative sampling')
parser.add_argument('--num_steps', type=int, default=20, help='Number of steps for negative sampling')
parser.add_argument('--stride', type=int, default=4, help='Stride for negative sampling steps')
parser.add_argument('--sample_start', type=int, default=10, help='Starting step for negative sampling')
parser.add_argument('--diffusion_loss_weight', type=float, default=0.000001, help='Weight for diffusion loss')
parser.add_argument('--temp', type=float, default=2, help='Temperature parameter for contrastive learning')
parser.add_argument('--positive_noise', type=float, default=0.01, help='Noise strength for positive samples in contrastive learning')

# Model mixing related parameters
args = parser.parse_args()
