import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import numpy as np
import math
from transformers import AutoTokenizer

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 2, emb_size: int = 768, img_size: int = 16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# MLP부분
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size))

class ImgEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 2,
                 emb_size: int = 768,
                 img_size: int = 16,
                 depth: int = 12,
                 num_tokens: int = 4,
                 **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
        )
        self.patchEmbedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.encoder = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.classicationhead = ClassificationHead(emb_size,num_tokens)
    def forward(self, img):
        feature = self.net(img)
        out = self.classicationhead(feature)


        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Decoder(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_decoder_layers, dropout_p):
        super().__init__()

        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, features, captions, tgt_mask):
        captions = self.embedding(captions) * math.sqrt(self.dim_model)
        captions = self.positional_encoder(captions)
        features = features.permute(1, 0, 2)
        captions = captions.permute(1, 0, 2)
        transformer_out = self.TransformerDecoder(captions, features, tgt_mask=tgt_mask)
        out = self.out(transformer_out)

        return out

class CT2captionModel(nn.Module):
    def __init__(self, num_tokens: int = 512, dim_model: int = 768, num_heads: int = 8, num_decoder_layers: int = 6,
                 dropout_p: int = 0.1, in_channels: int = 1, patch_size: int = 2, img_size: int = 16, depth: int = 12):
        super(CT2captionModel, self).__init__()
        self.encoder = ImgEncoder(in_channels, patch_size, dim_model, img_size, depth,num_tokens)
        self.decoder = Decoder(num_tokens, dim_model, num_heads, num_decoder_layers, dropout_p)

        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(dim_model, num_tokens)
        self.label_out = nn.Linear(dim_model, num_tokens)

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=depth,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

        self.label_transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=depth,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
    def forward(self, img, caption, label, tgt_mask, label_mask):
        features = self.encoder(img)
        #features = features.permute(1, 0, 2)

        labels = self.embedding(label) * math.sqrt(self.dim_model)
        labels = self.positional_encoder(labels)
        labels = labels.permute(1, 0, 2)

        label_feature = features+labels

        transformer_label_out = self.label_transformer(label_feature, labels, tgt_mask=label_mask)
        label_outputs = self.label_out(transformer_label_out)

        captions = self.embedding(caption) * math.sqrt(self.dim_model)
        captions = self.positional_encoder(captions)

        captions = captions.permute(1, 0, 2)
        transformer_out = self.transformer(transformer_label_out, captions, tgt_mask=tgt_mask)


        outputs = self.out(transformer_out)


        return label_outputs,outputs

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)

    def example_images(self,oimg, img, start_token, device, max_length=50, dim_model=64):
        result_caption = []
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        with torch.no_grad():
            x = self.encoder(img)
            y_input = start_token.unsqueeze(0)
            self.dim_model = dim_model

            #x = x.permute(1, 0, 2)

            for _ in range(max_length):

                tgt_mask = self.get_tgt_mask(y_input.size(1)).to(device)

                captions = self.embedding(y_input) * math.sqrt(self.dim_model)
                captions = self.positional_encoder(captions)
                captions = captions.permute(1, 0, 2)
                captions = captions.squeeze(dim=1)
                transformer_out = self.transformer(x, captions, tgt_mask=tgt_mask)
                predicted = self.out(transformer_out)

                next_item = predicted.topk(1)[1].view(-1)[-1].item()
                result_caption.append(next_item)

                next_item = torch.tensor([next_item], dtype=torch.long, device=device)

                y_input = y_input.squeeze(0)
                y_input = torch.cat((y_input, next_item)).unsqueeze(0)

                # Stop if model predicts end of sentence
                if predicted.topk(1)[1].view(-1)[-1].item() == 102:
                    break

        return tokenizer.decode(result_caption)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CT2captionModel().to(device)
    img = torch.rand(2, 1, 16, 16).to(device)
    caption = torch.rand(2, 110, ).to(device)

    out = model(img, caption)

    print(out)
    # summary(model, input_size=[(1, 16, 16),(110,)])

