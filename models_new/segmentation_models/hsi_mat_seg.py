import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from models import segmentation_models


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
            heads,
            n_filters,
            use_materialdb=False
    ):
        super().__init__()
        self.num_heads = heads
        self.n_filters = n_filters
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, n_filters * heads, bias=False)

        # use MLP to build a small path.
        if use_materialdb:
            self.mlp1 = Mlp(n_filters+8, 64, 128)
        else:
            self.mlp1 = Mlp(n_filters, 64, 128)
        self.mlp2 = Mlp(128)

    def forward(self, x_in, materialdb_info=None):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x_in = self.avg_pool(x_in)            
        x_in = x_in.permute([0, 2, 3, 1])
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q = self.to_q(x)
        k = self.to_k(x)

        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn - attn.detach().amin(dim=-1, keepdim=True)
        attn = attn / (attn.detach().amax(dim=-1, keepdim=True) - attn.detach().amin(dim=-1, keepdim=True))
        x = attn @ x.transpose(-2, -1)  # b,heads,d,hw
        x = x.permute(0, 2, 1)  # Transpose
        if materialdb_info is not None:
            materialdb_info = self.avg_pool(materialdb_info)
            materialdb_info = materialdb_info.permute(0, 2, 3, 1)
            x = torch.cat([x, materialdb_info.reshape(b, h*w, -1)], dim=2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = x.reshape(b, h, w, -1)
        x = x.permute([0, 3, 1, 2])

        return x, attn


class HSISegModel(pl.LightningModule):
    def __init__(self, spectral_filter_number, backbone_checkpoint, backbone_model, image_size, segment_classes, use_materialdb):
        super(HSISegModel, self).__init__()
        # use material db to refine the recovered hsi
        if use_materialdb:
            import json
            import pickle
            self.use_materialdb = True
            self.materialdb_des = torch.nn.parameter.Parameter(torch.tensor(pickle.load(open("data/spectraldb/properties.pkl", "rb"))), requires_grad=False)
            materialdb_sce = torch.tensor(pickle.load(open("data/spectraldb/sce.pkl", "rb")) / 100)
            materialdb_sce_rescaled_max = materialdb_sce.max(dim=1).values.unsqueeze(1)
            materialdb_sce_rescaled_min = materialdb_sce.min(dim=1).values.unsqueeze(1)
            self.standardised_materialdb_sce = torch.nn.parameter.Parameter(materialdb_sce - materialdb_sce_rescaled_min, requires_grad=False)
            self.materialdb_sce_range = torch.nn.parameter.Parameter(materialdb_sce_rescaled_max - materialdb_sce_rescaled_min, requires_grad=False)
            # self.materialdb_sci = pickle.load(open("data/materialdb/sci.pkl", "rb"))  
        else:
            self.use_materialdb = False

        self.filters = SpectralFilterModule(n_filters=spectral_filter_number, dim_head=31, dim=31, heads=1, use_materialdb=use_materialdb)
        self.segnet = segmentation_models.get_models(model_name="DBAT", backbone_checkpoint=backbone_checkpoint, image_size=image_size,
                                                     classes=segment_classes, backbone_model=backbone_model)
        
        
    def forward(self, hsi, image): 
        if self.use_materialdb:
            B, C, H, W = hsi.shape
            hsi_channle_last = hsi.permute(0, 2, 3, 1)     
            materialdb_info = torch.zeros(B, H, W, 8, device=image.device)
            # repeat the sce to work with direct operation, once per W
            for batch in range(B):
                per_sample_hsi = hsi_channle_last[batch,:,:,:]
                for row in range(H):
                    per_row_hsi = per_sample_hsi[row, :, :]
                    max_hsi_row = per_row_hsi.max(dim=1).values
                    min_hsi_row = per_row_hsi.min(dim=1).values
                    
                    # scale the range
                    k = max_hsi_row - min_hsi_row
                    materialdb_sce_rescaled = k.unsqueeze(1).unsqueeze(2) * self.standardised_materialdb_sce.unsqueeze(0) / self.materialdb_sce_range.unsqueeze(0) + min_hsi_row.unsqueeze(1).unsqueeze(2)
                    
                    # find the min diff idx
                    hsi_diff = materialdb_sce_rescaled - per_row_hsi.unsqueeze(1)
                    hsi_diff_abs_sum_pos = hsi_diff.abs().sum(dim=2).argmin(dim=1)

                    # append idx parameter coding
                    materialdb_info[batch, row, :, :] = self.materialdb_des[hsi_diff_abs_sum_pos]

                    # for col in range(W):
                    #     per_pixel_hsi = per_row_hsi[col, :]
                    #     max_hsi_pixel = per_pixel_hsi.max()
                    #     min_hsi_pixel = per_pixel_hsi.min()
                    
                    #     # scale the range
                    #     materialdb_sce_rescaled_max = self.materialdb_sce.max(dim=1).values.unsqueeze(1)
                    #     materialdb_sce_rescaled_min = self.materialdb_sce.min(dim=1).values.unsqueeze(1)
                    #     materialdb_sce_rescaled = (max_hsi_pixel - min_hsi_pixel) * (self.materialdb_sce - materialdb_sce_rescaled_min) / (materialdb_sce_rescaled_max - materialdb_sce_rescaled_min) + min_hsi_pixel
                        
                    #     # find the min diff idx
                    #     hsi_diff = materialdb_sce_rescaled - per_pixel_hsi
                    #     hsi_diff_abs_sum_pos = hsi_diff.abs().sum(1).argmin()
                    #     # append idx parameter coding
                    #     materialdb_des_pixel = list(self.materialdb_des[hsi_diff_abs_sum_pos]["properties"].values())
                    #     materialdb_info[batch, row, col, :] = torch.tensor([materialdb_des_pixel[0], materialdb_des_pixel[1], materialdb_des_pixel[2], materialdb_des_pixel[6], materialdb_des_pixel[7], materialdb_des_pixel[8], materialdb_des_pixel[9], materialdb_des_pixel[10]])

            materialdb_info = materialdb_info.permute(0, 3, 1, 2)
        else:
            materialdb_info = None
        filtered_multispectral, filters = self.filters(hsi, materialdb_info)  # second channel is the number of filters
        # multi_spectral_x = torch.concat((image, filtered_multispectral), dim=1)  # first 3 channels are the RGB images
        image = image - 0.5
        pred_mask = self.segnet(image, filtered_multispectral)
        return pred_mask, filters
