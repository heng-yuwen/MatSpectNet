import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from models import segmentation_models
import time
import os


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectralFilterModule(nn.Module):
    """Predict n_filters spectral filters to mimic response curve."""
    def __init__(
            self,
            dim,
            dim_head,
            heads
    ):

        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        # Section 3.4, create the filter
        out = out.mean(dim=(0, 1, 2))
        min_v = out.min()
        max_v = out.max()

        out = (out - min_v) / (max_v - min_v)

        return out

        # x_in = self.avg_pool(x_in)            
        # x_in = x_in.permute([0, 2, 3, 1])
        # b, h, w, c = x_in.shape
        # x = x_in.reshape(b, h * w, c)
        # q = self.to_q(x)
        # k = self.to_k(x)

        # # q: b,heads,hw,c
        # q = q.transpose(-2, -1)
        # k = k.transpose(-2, -1)
        # q = F.normalize(q, dim=-1, p=2)
        # k = F.normalize(k, dim=-1, p=2)
        # attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        # attn = attn - attn.detach().amin(dim=-1, keepdim=True)
        # attn = attn / (attn.detach().amax(dim=-1, keepdim=True) - attn.detach().amin(dim=-1, keepdim=True))
        # x = attn @ x.transpose(-2, -1)  # b,heads,d,hw
        # x = x.permute(0, 2, 1)  # Transpose
        # if materialdb_info is not None:
        #     materialdb_info = self.avg_pool(materialdb_info)
        #     materialdb_info = materialdb_info.permute(0, 2, 3, 1)
        #     x = torch.cat([x, materialdb_info.reshape(b, h*w, -1)], dim=2)
        # x = self.mlp1(x)
        # x = self.mlp2(x)
        # x = x.reshape(b, h, w, -1)
        # x = x.permute([0, 3, 1, 2])


class HSISegModelV2(pl.LightningModule):
    def __init__(self, spectral_filter_number, backbone_checkpoint, backbone_model, image_size, segment_classes, use_materialdb):
        super(HSISegModelV2, self).__init__()
        # use material db to refine the recovered hsi
        if use_materialdb:
            import json
            import pickle
            import numpy as np
            self.use_materialdb = True
            self.mlp1 = Mlp(spectral_filter_number+8, 64, 128)
   
            self.materialdb_des = torch.nn.parameter.Parameter(torch.tensor(pickle.load(open("data/spectraldb/properties.pkl", "rb"))), requires_grad=False)
            materialdb_sce = np.load("data/spectraldb/shape_metrics.npy")
            import faiss 
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(materialdb_sce.shape[1])
            # self.gpu_index = faiss.index_cpu_to_gpu(res, int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0, index_flat)
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
            # self.gpu_index = faiss.index_cpu_to_all_gpus(index_flat)
            self.gpu_index.add(materialdb_sce)
            # print("There are {} vectors in the database!".format(self.gpu_index.ntotal))
        else:
            self.use_materialdb = False
            self.mlp1 = Mlp(n_filters, 64, 128)
        
        self.mlp2 = Mlp(128)
        self.filters = nn.Sequential(*[SpectralFilterModule(dim_head=31, dim=31, heads=1) for _ in range(spectral_filter_number)])
        self.spectral_filter_number = spectral_filter_number
        self.segnet = segmentation_models.get_models(model_name="DBAT", backbone_checkpoint=backbone_checkpoint, image_size=image_size,
                                                     classes=segment_classes, backbone_model=backbone_model)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
    def forward(self, hsi, image): 
        if self.use_materialdb:
            B, C, H, W = hsi.shape
            hsi_channle_last = hsi.permute(0, 2, 3, 1) # channel last 
            # materialdb_info = torch.zeros(B, H, W, 8, device=image.device)
            hsi_queries = hsi_channle_last.detach().reshape(-1, C)

            with torch.no_grad():
                hsi_shape_queries = torch.zeros(hsi_queries.shape[0], 465, device=hsi_queries.device)

                # start_time = time.time()
                length = 30
                end = 30
                for i in range(1, 31):
                    hsi_shape_queries[:, end-length:end] = torch.abs(hsi_queries[:, i-1:30] - hsi_queries[:, i:])
                    length -= 1
                    end += length
                # end_time = time.time()

                # find the best match
                D, I = self.gpu_index.search(hsi_shape_queries.detach().cpu().numpy(), 1)
                I = I.reshape(-1)
                materialdb_info = self.materialdb_des[I].reshape(B, H, W, -1)

                # materialdb_info = materialdb_info.permute(0, 3, 1, 2)
        else:
            materialdb_info = None

        # Section 3.3 and 3,4

        filters = [filter_learner(hsi_channle_last) for filter_learner in self.filters]  # second channel is the number of filters
        filters = torch.stack(filters, dim=1)
        filtered_multispectral = torch.matmul(hsi_channle_last, filters) # channel first
        multi_modal_hsi = torch.concat((filtered_multispectral, materialdb_info), dim=3)
        multi_modal_hsi = self.avg_pool(multi_modal_hsi.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # MLP in Figure 1
        multi_modal_hsi = self.mlp1(multi_modal_hsi)
        multi_modal_hsi = self.mlp2(multi_modal_hsi)

        image = image - 0.5 # normalise to [-0.5, 0.5]
        pred_mask = self.segnet(image, multi_modal_hsi.permute(0, 3, 1, 2))
        return pred_mask, filters
