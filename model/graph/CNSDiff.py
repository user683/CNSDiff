import torch
import os
import json
import pandas as pd
from torch.distributions import Beta
from conf.params import *  
from module.MLP_model import MLP
from module.diffusion_model import *
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss, contrastive_loss

device = torch.device(f'cuda:{args.gpu}')


class CNSDiff(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CNSDiff, self).__init__(conf, training_set, test_set)
        yaml = self.config['CNSDiff']
        self.n_layers = int(yaml['n_layer'])
        self.pre_train_item_emb = None
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

        # Initialize other parameters
        self.Diffusion = DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min,
                                          args.noise_max, args.steps, device).to(device)

        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.MLP = MLP(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

        # Initialize optimizer
        self.MLP_opt = torch.optim.Adam([{'params': self.MLP.parameters(), 'weight_decay': 0}], lr=args.lr)

        # Get all items
        # Save metrics for each epoch
        from collections import defaultdict
        self.result = defaultdict(list)

    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        # Initial Beta parameters - let random sampling dominate (e.g., 90%)
        initial_alpha, initial_beta = args.initial_alpha, args.initial_beta   # 2 and 8 are suitable ratios for yelp 0.5 0.9
        # Final Beta parameters - let generated sampling take larger proportion (e.g., 50% each)
        final_alpha, final_beta = args.final_alpha, args.final_beta    #1 9 
        
        cl_weight = args.cl_weight  # How about changing this ratio to 1, original 0.5
        min_mix_ratio = args.min_mix_ratio # Ensure at least 10% random sampling  # yelp2018 value is 0

        for epoch in range(self.maxEpoch):
            # Dynamically adjust alpha and beta - linear interpolation
            progress = epoch / self.maxEpoch  # Training progress [0,1]
            alpha = initial_alpha + (final_alpha - initial_alpha) * progress
            beta = initial_beta + (final_beta - initial_beta) * progress
            
            model.train()

        
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                # Data preparation
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rand_neg_samples = neg_item_emb

                # Mixed ratio sampling 
                mix_ratio = Beta(alpha, beta).sample((len(user_idx), 1)).to(pos_item_emb.device)
                mix_ratio = torch.clamp((1 - min_mix_ratio) * mix_ratio + min_mix_ratio, 0, 1)

                terms = self.Diffusion.caculate_losses(self.MLP, pos_item_emb, args.reweight)

                # Systematic negative sampling 
                with torch.no_grad():
                    num_steps = args.num_steps
                    stride = args.stride
                    sample_steps = list(range(args.sample_start, num_steps + 1, stride))

                    all_neg_samples = []
                    scores = []

                    # Ensure base tensors are on correct device
                    device = pos_item_emb.device  # Get main device
                    user_emb = user_emb.to(device)
                    
                    for step in sample_steps:
                        step_neg_emb = self.Diffusion.p_sample(
                            self.MLP, pos_item_emb, step, False
                        ).to(device)  # Ensure output is on correct device
                        
                        step_scores = torch.sigmoid(torch.sum(user_emb * step_neg_emb, dim=1))
                        all_neg_samples.append(step_neg_emb)
                        scores.append(step_scores)

                    # Confirm device consistency before stacking
                    all_neg_samples = torch.stack(all_neg_samples).to(device)
                    scores = torch.stack(scores).to(device)
                    
                    hard_neg_idx = torch.argmax(scores, dim=0)
                    hard_neg_samples = all_neg_samples[hard_neg_idx, torch.arange(len(hard_neg_idx), device=device)]  # Ensure index device consistency


                # Mix negative samples
                mixed_neg = mix_ratio * hard_neg_samples + (1 - mix_ratio) * rand_neg_samples
                # mixed_neg = hard_neg_samples

                cl_loss = contrastive_loss(
                    anchor=pos_item_emb,
                    positive=pos_item_emb + args.positive_noise * torch.randn_like(pos_item_emb),
                    negative=torch.cat([hard_neg_samples, rand_neg_samples], dim=0),
                    # negative = hard_neg_samples,
                    temp=args.temp
                )

                bpr_loss = -torch.log(torch.sigmoid(
                    torch.sum(user_emb * (pos_item_emb - mixed_neg), dim=1)
                )).mean()

                total_loss = (
                    bpr_loss  +
                    + cl_weight * cl_loss
                    + args.diffusion_loss_weight*terms["loss"].mean()  #  1*10^-5
                    + l2_reg_loss(
                        self.reg,
                        model.embedding_dict['user_emb'][user_idx],
                        model.embedding_dict['item_emb'][pos_idx],
                        model.embedding_dict['item_emb'][neg_idx]
                    ) / self.batch_size
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()



class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
