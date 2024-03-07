import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from sklearn.cluster import KMeans

from drnn import *


class ConcatFeas(nn.Module):

    def __init__(self, dim_inp, dim_out):
        super(ConcatFeas, self).__init__()

        self.dim_inp = dim_inp
        self.dim_out = dim_out
        self.joint = nn.Linear(2*self.dim_inp, self.dim_out)

    def forward(self, f1, f2):
        new_f = torch.cat([f1, f2], dim=-1)
        new_f = self.joint(new_f)
        return new_f

    
class Model_DTSSOCC(nn.Module):
    
    def __init__(self, args):

        super(Model_DTSSOCC, self).__init__()
        self.args = args
        self.input_dims = args.dim
        self.dilations = self.args.dilations
        self.hidden_structs = [self.args.hidden_size]*len(self.dilations)

        self.drnn = multi_dRNN_with_dilations(args, self.hidden_structs, self.dilations, self.input_dims).to(self.args.device)
        self.parms = list(self.drnn.parameters())

        self.fea_joint = ConcatFeas(self.args.hidden_size, self.args.hidden_size).to(self.args.device)
        self.parms_mlp = list(self.fea_joint.parameters())

        self.layer_indexes = list(range(0,len(self.args.nums_cluster_each_layer)))
        
        self.centers = []
        self.tranforms = []
        for i in self.layer_indexes:
            parm = (torch.rand(self.args.nums_cluster_each_layer[i], self.args.hidden_size) - 0.5) * 2
            parm = parm.to(self.args.device)
            parm.requires_grad = True
            self.parms += list([parm])
            self.centers.append(parm)

            parm = nn.Linear(self.args.hidden_size, self.args.hidden_size).to(self.args.device)
            self.parms += list(parm.parameters())
            self.tranforms.append(parm)
            
        self.out_proj = {}
        for j in range(len(self.dilations)):
            self.out_proj[j] = nn.Linear(self.args.hidden_size, self.input_dims).to(self.args.device)
            self.parms += list(self.out_proj[j].parameters())
            
        self.init_optimizer()
        
        self.cosine_sim3 = nn.CosineSimilarity(dim=3)
        self.cosine_sim4 = nn.CosineSimilarity(dim=4)
        
        self.alpha = self.args.alpha
        self.criterion_mse = nn.MSELoss()
        self.center_smooth = 0.5
        self.training_epoch = 0
        
    def this(self):
        return self
        
    def init_optimizer(self, beta_lower=0.1):
        
        self.optimizer = optim.Adam(self.parms, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(beta_lower, 0.999))   
        self.optimizer_mlp = optim.Adam(self.parms_mlp, lr=0.00001, weight_decay=self.args.weight_decay, betas=(beta_lower, 0.999))
        
    def assignment(self, Hs, Cj, _tranform, importance=None):

        # >>> Hs [bsz, seq_len, num_inp_vectors, fea_dim] 
        # >>> Cj torch.Size([12, 32])
        num_centers = np.shape(Cj)[0]
        dim_feas = np.shape(Cj)[1]
        seq_len = np.shape(Hs)[1]
        num_feas = np.shape(Hs)[2]
            
        F_t = Hs  # [bsz, seq_len, num_inp_vectors, fea_dim]
        Hs = Hs.unsqueeze(3).repeat(1,1,1,num_centers,1)  # torch.Size([bsz, seq_len, num_inp_vectors, num_clusters, fea_dim])
        Cj = Cj.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1,1,num_feas,1,1)  # torch.Size([1, 1, num_inp_vectors, num_clusters, fea_dim])
        
        dists = 0.5 * (1 - self.cosine_sim4(Hs, Cj))  # torch.Size([bsz, seq_len, num_inp_vectors, num_clusters])
        assighted_weights_3 = F.softmax(-self.alpha * dists, dim=3)  # softmax for num_clusters
        assighted_weights = F.softmax(-self.alpha * dists, dim=2)    # softmax for num_inp_vectors 
        
        assighted_weights_3 = assighted_weights_3.permute(0,1,3,2)  # torch.Size([bsz, seq_len, num_clusters, num_inp_vectors]) 
        assighted_weights = assighted_weights.permute(0,1,3,2)      # torch.Size([bsz, seq_len, num_clusters, num_inp_vectors]) 
        
        F_tj = F.relu(_tranform(F_t))  # [bsz, seq_len, num_inp_vectors, fea_dim]
        assighted_weights_3 = assighted_weights_3.reshape([-1, num_centers, num_feas])  # torch.Size([-1, num_clusters, num_inp_vectors])
        assighted_weights = assighted_weights.reshape([-1, num_centers, num_feas])      # torch.Size([-1, num_clusters, num_inp_vectors])
        
        if importance is None:
            importance = F.softmax(torch.sum(assighted_weights_3, dim=2), dim=1)  # torch.Size([seq_len, 12])
        else:
            # torch.Size([bsz, seq_len, num_clusters])
            importance = importance.reshape([-1, num_feas]).unsqueeze(2)  # torch.Size([bsz*seq_len, num_inp_vectors, 1])
            importance = F.softmax(assighted_weights_3.bmm(importance).squeeze(2), dim=1) # torch.Size([bsz*seq_len, num_clusters, 1])
        
        importance = importance.reshape([-1, seq_len, num_centers])  # torch.Size([bsz, seq_len, num_clusters])

        F_tj = F_tj.reshape([-1, num_feas, dim_feas])  # torch.Size([bsz, num_inp_vectors, fea_dim])
        f_t_new = assighted_weights.bmm(F_tj)  # L*L-1  # torch.Size([bsz, num_clusters, fea_dim])
        f_t_new = f_t_new.reshape([-1, seq_len, num_centers, dim_feas])  # torch.Size([bsz, seq_len, num_clusters, fea_dim])
 
        return f_t_new, importance
        
    def forward(self, data_x):

        # np.shape(data_x): torch.Size([bsz, seq_len, fea_dim])
        reshape_data_x = data_x.permute(1,0,2)      # torch.Size([seq_len, bsz, fea_dim])
        drnn_outs = self.drnn.multi_dRNN(reshape_data_x)    # drnn_outs: torch.Size([num_layer, seq_len, bsz, fea_dim])
        drnn_outs = drnn_outs.permute(0,2,1,3)          # drnn_outs: torch.Size([num_layer, bsz, seq_len, fea_dim])

        batch_size, seq_len = np.shape(drnn_outs)[1], np.shape(drnn_outs)[2]
        
        H = drnn_outs   # torch.Size([num_layer, bsz, seq_len, fea_dim])

        importance = None
        Rs = []

        for i, j in enumerate(self.layer_indexes):
            
            Hs = H[j].unsqueeze(2) 
            # torch.Size([bsz, seq_len, 1, fea_dim])
            if i > 0:
                num_clusters = np.shape(updated_feas)[2]
                Hs = Hs.repeat(1,1,num_clusters,1)
                Hs = self.fea_joint(Hs, updated_feas)
            
            Cj = self.centers[j]  # [num_clusters, dim_fea]
            tranform = self.tranforms[j]
            updated_feas, importance = self.assignment(Hs, Cj, tranform, importance)
            Rs.append(importance.detach().cpu().numpy())

        final_feas = updated_feas       # torch.Size([bsz, seq_len, num_classes, fea_dim])
        
        return drnn_outs, final_feas, Rs, importance

    def estimate_cluster_centers(self, data_x):

        reshape_data_x = data_x.permute(1,0,2)      # torch.Size([seq_len, bsz, fea_dim])
        drnn_outs = self.drnn.multi_dRNN(reshape_data_x)    # drnn_outs: torch.Size([num_layer, seq_len, bsz, fea_dim])
        drnn_outs = drnn_outs.permute(0,2,1,3)          # drnn_outs: torch.Size([num_layer, bsz, seq_len, fea_dim])

        batch_size, seq_len = np.shape(drnn_outs)[1], np.shape(drnn_outs)[2]

        H = drnn_outs   # torch.Size([num_layer, bsz, seq_len, fea_dim])

        importance = None
        Rs = []
        
        local_centers = []

        for i, j in enumerate(self.layer_indexes):
                
            datas = drnn_outs[j]

            Hs = H[j].unsqueeze(2)  
            # torch.Size([bsz, seq_len, 1, fea_dim])
            if i > 0:
                num_clusters = np.shape(updated_feas)[2]
                Hs = Hs.repeat(1,1,num_clusters,1)
                Hs = self.fea_joint(Hs, updated_feas)
            
            Cj = self.centers[j]  # [num_clusters, dim_fea]
            tranform = self.tranforms[j]
            updated_feas, importance = self.assignment(Hs, Cj, tranform, importance)

            if i == 0:
                data_i = datas.reshape(-1, np.shape(datas)[-1]).cpu().data.numpy()
            else:
                updated_feas_np = torch.reshape(updated_feas, (-1, np.shape(updated_feas)[-1]))
                data_i = updated_feas_np.cpu().data.numpy()

            print(">>> estimate cluster layer {} at the rnn scale {}".format(i, j), np.shape(data_i))

            kmeans = KMeans(n_clusters=self.args.nums_cluster_each_layer[j]).fit(data_i)
            centers = kmeans.cluster_centers_
            local_centers.append(centers)            
            
        return local_centers

    def initialize_cluster_centers(self, centers):
        for i in range(0, len(centers)):
            self.centers[i].data = torch.from_numpy(centers[i]).to(self.args.device)
    
    def get_dtssocc_loss(self, drnn_outs, final_feas, Rs, final_importance, data_y, reduce_=True):
                      
        # print(data_y.shape) torch.Size([32, 100])
        # print(scores.shape) torch.Size([32, 100, 6])
        # print(centers.shape) torch.Size([1, 1, 6, 64])
        # print(final_feas.shape) torch.Size([32, 100, 6, 64])
        # print(final_importance.shape) torch.Size([32, 100, 6])
        # print(data_y.shape) torch.Size([32, 100])
        
        centers = self.centers[self.layer_indexes[-1]].unsqueeze(0).unsqueeze(1)  # centers torch.Size([1, 1, num_classes, fea_dim])         
        scores = 0.5 * (1 - self.cosine_sim3(final_feas, centers))  # centers torch.Size([bsz, seq_len, num_classes])  
        
        #for i in range(0, scores.shape[0]):
        #    for j in range(0, scores.shape[1]):
        #        if data_y[i][j] == -1:
        #            for k in range(0, scores.shape[2]):
        #                scores[i][j][k] = 1/scores[i][j][k]
                        
        scores = final_importance * scores  
        if reduce_:
            scores = torch.mean(scores, dim=2) # torch.Size([bsz, seq_len]) torch.Size([32, 100])
            scores = torch.mean(scores)
     
        return scores

    def get_orth_loss(self):

        # ==========================================================
        # get orthogonality penalty: P = (CCT - I)
        # ==========================================================
        loss_orth = []
        for j in range(len(self.args.dilations)):
            
            # beta * (||W.W.T * (1-I)||_F)^2
            if self.args.nums_cluster_each_layer[j] > 1:
                I = torch.eye(self.args.nums_cluster_each_layer[j]).to(self.args.device)
                Cj = self.centers[j]  # num_classes, fea_dim
                CCT = Cj @ Cj.transpose(0, 1)
                loss = torch.mean((CCT - I) ** 2)
                loss_orth.append(loss)

        loss_orth = torch.stack(loss_orth, dim=0)
        loss_orth = torch.mean(loss_orth)

        return loss_orth

    def get_tss_loss(self, data_x, drnn_outs):

        # ==========================================================
        # prediction loss
        # ==========================================================
        teachers = data_x  # torch.Size([bsz, seq_len, fea_dim])

        pred_losses = []
        for i in range(len(self.args.dilations)):

            dilation = self.args.dilations[i]
            outputs_i = self.out_proj[i](drnn_outs[i])

            pred_loss = self.criterion_mse(outputs_i[:,:-dilation,:], teachers[:,dilation:,:])
            pred_losses.append(pred_loss)
        
        pred_losses = torch.stack(pred_losses, dim=0)
        loss_tss = torch.mean(pred_losses)

        return loss_tss

    def gradient_update(self, loss):

        self.optimizer.zero_grad()
        self.optimizer_mlp.zero_grad()

        loss.backward()
        
        self.optimizer.step()
            
        torch.nn.utils.clip_grad_norm_(self.parms_mlp, 0.1)
        self.optimizer_mlp.step()    
