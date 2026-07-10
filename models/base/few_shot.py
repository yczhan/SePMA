import torch
from torch.functional import norm
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations
from torch.nn.init import xavier_normal_ 
import numpy as np
# from torch.nn.modules.activation import MultiheadAttention

from torch.autograd import Variable
import torchvision.models as models
from einops import rearrange
import os
from torch.autograd import Variable

from utils.registry import Registry
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY
# from collections import OrderedDict
from typing import Tuple, Union
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

# from .model import build_model
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = SimpleTokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, cfg, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, spatial=False):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result





class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, spatial=False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spatial = spatial

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # if self.spatial:
        #     x_copy = x[1:]
        # x, _ = F.multi_head_attention_forward(
        #     query=x[:1], key=x, value=x,
        #     embed_dim_to_check=x.shape[-1],
        #     num_heads=self.num_heads,
        #     q_proj_weight=self.q_proj.weight,
        #     k_proj_weight=self.k_proj.weight,
        #     v_proj_weight=self.v_proj.weight,
        #     in_proj_weight=None,
        #     in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        #     bias_k=None,
        #     bias_v=None,
        #     add_zero_attn=False,
        #     dropout_p=0,
        #     out_proj_weight=self.c_proj.weight,
        #     out_proj_bias=self.c_proj.bias,
        #     use_separate_proj_weight=True,
        #     training=self.training,
        #     need_weights=False
        # )
        if self.spatial:
            if self.spatial=="v2":
                cls_token, _ = F.multi_head_attention_forward(
                            query=x[:1], key=x, value=x,
                            embed_dim_to_check=x.shape[-1],
                            num_heads=self.num_heads,
                            q_proj_weight=self.q_proj.weight,
                            k_proj_weight=self.k_proj.weight,
                            v_proj_weight=self.v_proj.weight,
                            in_proj_weight=None,
                            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                            bias_k=None,
                            bias_v=None,
                            add_zero_attn=False,
                            dropout_p=0,
                            out_proj_weight=self.c_proj.weight,
                            out_proj_bias=self.c_proj.bias,
                            use_separate_proj_weight=True,
                            training=self.training,
                            need_weights=False
                            )
                x_mid = self.v_proj(x[1:])
                x_mid = self.c_proj(x_mid)
                x = torch.cat([cls_token, x_mid], dim=0)
                return x.squeeze(0)
            else:

                x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
                )
                # return torch.cat([x, self.c_proj(x_copy)], dim=0)
                return x.squeeze(0)
        else:
            x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
            )
            return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, spatial=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, spatial)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 spatial=False,
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,   # (3, 4, 6, 3)
                output_dim=embed_dim,   # 1024
                heads=vision_heads,     # 32
                input_resolution=image_resolution,    # 224
                width=vision_width,      # 64
                spatial=spatial
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, spatial=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, spatial=spatial,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size, groups=1)
            self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# MODEL_REGISTRY = Registry("Model")
# STEM_REGISTRY = Registry("Stem")
# BRANCH_REGISTRY = Registry("Branch")
# HEAD_REGISTRY = Registry("Head")
# HEAD_BACKBONE_REGISTRY = Registry("HeadBackbone")

class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q

class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention_qkv(dim, Attention_qkv(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x

class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x




class Attention_qkv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)    # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    def __init__(self, cfg):
        super(CNN_FSHead, self).__init__()
        args = cfg
        self.train()
        self.args = args

        last_layer_idx = -1
        
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet18":
            backbone = models.resnet18(pretrained=True) 
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

        elif self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet34":
            backbone = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

        elif self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.TRAIN.DDP_GPU > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.TRAIN.DDP_GPU)])
    
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
        
        


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


@HEAD_REGISTRY.register()
class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, cfg, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
        # temporal_set_size=3
        args = cfg
        
        self.args = args
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.trans_linear_in_dim = 2048
        else:
            self.trans_linear_in_dim = 512
        # self.trans_linear_in_dim = 2048
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.DATA.NUM_INPUT_FRAMES * 1.5)
        self.pe = PositionalEncoding(self.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.DATA.NUM_INPUT_FRAMES)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]   # [35, 8, 2048]
        n_support = support_set.shape[0]   # [5, 8, 2048]
        
        # static pe
        support_set = self.pe(support_set)   # [5, 8, 2048]
        queries = self.pe(queries)      # queries

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)   # [5, 28, 4096]
        queries = torch.stack(q, dim=-2)    # [35, 28, 4096]

        # apply linear maps
        support_set_ks = self.k_linear(support_set)   # [5, 28, 1152]
        queries_ks = self.k_linear(queries)           # [35, 28, 1152]
        support_set_vs = self.v_linear(support_set)   # [5, 28, 1152]
        queries_vs = self.v_linear(queries)           # [35, 28, 1152]
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)   # [0., 1., 2., 3., 4.]

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.TRAIN.WAY, device=queries.device)   # [35, 5]

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))   # [1, 28, 1152]
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))   # [1, 28, 1152]
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)   # ([35, 1, 28, 1152], [1, 1152, 28]) --> [35, 1, 28, 28]

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)     # [35, 28, 1, 28]
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)     # [35, 28, 28]
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)      # [980, 28]
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)    # [35, 28, 1, 28]
            class_scores = class_scores.permute(0,2,1,3)       # [35, 1, 28, 28]
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)    # [35, 1, 28, 1152]
            query_prototype = torch.sum(query_prototype, dim=1)      # [35, 1, 28, 1152]
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype   # [35, 28, 1152]
            norm_sq = torch.norm(diff, dim=[-2,-1])**2    # [35]
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict


@HEAD_REGISTRY.register()
class CNN_TRX(CNN_FSHead):
    """
    Backbone connected to Temporal Cross Transformers of multiple cardinalities.
    """
    def __init__(self, cfg):
        super(CNN_TRX, self).__init__(cfg)
        args = cfg
        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [2,3]
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, inputs):
        # support_images, support_labels, target_images = inputs
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)





def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


@HEAD_REGISTRY.register()
class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_OTAM, self).__init__(cfg)
        args = cfg
        self.args = cfg

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]

        support_features, target_features = self.get_feats(support_images, target_images)
        # [25, 8, 2048] [25, 8, 2048]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query
        cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))


        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())





@HEAD_REGISTRY.register()
class CNN_CrossTransformer(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_CrossTransformer, self).__init__(cfg)
        args = cfg
        self.args = cfg
        self.dim = 2048
        # self.hidden_dim = 512  # v0 
        # self.hidden_dim = 2048   # v1
        self.hidden_dim = 1024   # v2
        self.way = cfg.TRAIN.WAY
        self.shot = cfg.TRAIN.SHOT
        self.key_head = nn.Conv1d(self.dim, self.hidden_dim, 1, bias=False)
        self.query_head = self.key_head
        self.value_head = nn.Conv1d(self.dim, self.hidden_dim, 1, bias=False)

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]
        # support_images, support_labels, target_images = inputs

        support_features, query_image_features = self.get_feats(support_images, target_images)
        # [25, 8, 2048] [25, 8, 2048]

        unique_labels = torch.unique(support_labels)
        support_features = [torch.index_select(support_features, 0, extract_class_indices(support_labels, c)) for c in unique_labels]
        support_features = torch.cat(support_features, 0)  # [25, 8, 2048]
     
        query = self.query_head(query_image_features.permute(0,2,1))   # [25, 512, 8]
        support_key = self.key_head(support_features.permute(0,2,1))
        support_value = self.value_head(support_features.permute(0,2,1))

        ## flatten pixels & k-shot in support (j & m in the paper respectively)
        support_key = support_key.view(self.way, self.shot, support_key.shape[1], -1)
        support_value = support_value.view(self.way, self.shot, support_value.shape[1], -1)

        support_key = support_key.permute(0, 2, 3, 1)
        support_value = support_value.permute(0, 2, 3, 1)

        support_key = support_key.contiguous().view(self.way, support_key.shape[1], -1)   # [5, 512, 40]
        support_value = support_value.contiguous().view(self.way, support_value.shape[1], -1)

		## v is j images' m pixels, ie k-shot*h*w
        attn_weights = torch.einsum('bdp,ndv->bnpv', query, support_key) * (self.hidden_dim ** -0.5)   # [15, 5, 8, 40]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
		
        ## get weighted sum of support values
        support_value = support_value.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)   # [15, 5, 512, 40]
        query_aligned_prototype = torch.einsum('bnpv,bndv->bnpd', attn_weights, support_value)   # [15, 5, 8, 512]

		### Step 3: Calculate query value
        query_value = self.value_head(query_image_features.permute(0,2,1)).permute(0,2,1)   # [25, 8, 512]
		# query_value = query_value.view(query_value.shape[0], -1, query_value.shape[1]) ##bpd
		
		### Step 4: Calculate distance between queries and supports
        distances = []
        for classid in range(query_aligned_prototype.shape[1]):
            support_features = rearrange(F.normalize(query_aligned_prototype[:, classid], dim=2), 'b s d -> b (s d)')     # [15, 4096]
            target_features = rearrange(F.normalize(query_value, dim=2), 'b s d -> b (s d)')      # [15, 4096]
            # dxc = torch.matmul(target_features, support_features.transpose(0,1))
            dxc = (target_features*support_features).sum(1)/8   # vo is no/8

			# dxc = torch.cdist(query_aligned_prototype[:, classid], 
			# 								query_value, p=2)
			# dxc = dxc**2
			# B,P,R = dxc.shape
			# dxc = dxc.sum(dim=(1,2)) / (P*R)
            distances.append(dxc)
		
        # class_dists = rearrange(class_dists, 'c q -> q c')
        class_dists = torch.stack(distances, dim=1)
        return_dict = {'logits': class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    


@HEAD_REGISTRY.register()
class CNN_TSN(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_TSN, self).__init__(cfg)
        args = cfg
        self.norm_sq_dist = False


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        support_features, target_features = self.get_feats(support_images, target_images)   # [25, 8, 2048] [25, 8, 2048]
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        if self.norm_sq_dist:
            class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)
            
            diffs = [target_features - class_prototypes[i] for i in unique_labels]
            diffs = torch.stack(diffs)

            norm_sq = torch.norm(diffs, dim=[-1])**2
            distance = - rearrange(norm_sq, 'c q -> q c')
            return_dict = {'logits': distance}

        else:
            class_sim = cos_sim(target_features, support_features)
            class_sim = [torch.mean(torch.index_select(class_sim, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
            class_sim = torch.stack(class_sim)
            class_sim = rearrange(class_sim, 'c q -> q c')
            return_dict = {'logits': class_sim}

        return return_dict


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class PositionalEncoder(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)


@HEAD_REGISTRY.register()
class CNN_HyRSM_1shot(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, cfg):
        super(CNN_HyRSM_1shot, self).__init__(cfg)
        last_layer_idx = -1
        self.args = cfg
        
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                   
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [35, 5, 8, 2048]  V1
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        support_features_ext = support_features.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
 
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists
        
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())





@HEAD_REGISTRY.register()
class CNN_HyRSM_5shot(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_5shot, self).__init__(cfg)
        args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        # self.relu1 = nn.ReLU(inplace=True)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY

        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        # feature_in = self.temporal_atte(feature_in, feature_in, feature_in)  # .view(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)) [35, 6, 2048]  45%
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        # frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        # calculate query -> support and support -> query
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 
        
        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits}
        return return_dict


@HEAD_REGISTRY.register()
class CNN_HyRSM_plusplus_1shot(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, cfg):
        super(CNN_HyRSM_plusplus_1shot, self).__init__(cfg)
        last_layer_idx = -1
        self.args = cfg
        
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50" or self.args.VIDEO.HEAD.BACKBONE_NAME == "inception_v3":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        elif hasattr(self.args.TRAIN,"NO_POSITION"):
            self.pe = nn.Sequential()
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        
        max_seq_len = self.args.DATA.NUM_INPUT_FRAMES
        temproal_regular = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular[i, j] = 1./(pow(i-j,2)+1.0)
                else:
                    temproal_regular[i, j] = 1.-torch.exp(torch.tensor(-pow(abs(i-j)-self.args.TRAIN.WINDOW_SIZE,2)/self.args.TRAIN.TEMPORAL_BALANCE))
        self.temproal_regular = temproal_regular.cuda()

        temproal_regular_label = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular_label[i, j] = 1.0
                
        self.temproal_regular_label = temproal_regular_label.cuda()
                

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        # h_dim = int(support_features.shape[2])
        # w_dim = int(support_features.shape[3])

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES
        
        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [35, 5, 8, 2048]  V1
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        # class_logits = self.classification_layer(torch.cat([support_features.mean(1), target_features.mean(1)], 0))
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None
        support_features_ext = support_features.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        dim = support_features.shape[-1]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        # frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        frame_sim_support = torch.matmul(F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_support*n_queries, frame_num, frame_num)
        frame_dists_support = 1 - frame_sim_support

        frame_sim_query = torch.matmul(F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_queries, frame_num, frame_num)
        frame_dists_query = 1 - frame_sim_query
        
        if hasattr(self.args.TRAIN, "BALANCE_COEFFICIENT") and self.args.TRAIN.BALANCE_COEFFICIENT:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        else:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        
        cum_dists_regular = dists
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits, "loss_temporal_regular": loss_temporal_regular}
        
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


@HEAD_REGISTRY.register()
class CNN_HyRSM_plusplus_5shot(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_plusplus_5shot, self).__init__(cfg)
        args = cfg
        # args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50" or self.args.VIDEO.HEAD.BACKBONE_NAME == "inception_v3":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        elif hasattr(self.args.TRAIN,"NO_POSITION"):
            self.pe = nn.Sequential()
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)

        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:            
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        

        max_seq_len = self.args.DATA.NUM_INPUT_FRAMES
        temproal_regular = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular[i, j] = 1./(pow(i-j,2)+1.0)
                else:
                    
                    temproal_regular[i, j] = 1.-torch.exp(torch.tensor(-pow(abs(i-j)-self.args.TRAIN.WINDOW_SIZE,2)/self.args.TRAIN.TEMPORAL_BALANCE))
        self.temproal_regular = temproal_regular.cuda()

        temproal_regular_label = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular_label[i, j] = 1.0
                
        self.temproal_regular_label = temproal_regular_label.cuda()

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY

        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)
        dim = support_features.shape[-1]

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        frame_sim_support = torch.matmul(F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_support*n_queries, frame_num, frame_num)
        frame_dists_support = 1 - frame_sim_support

        frame_sim_query = torch.matmul(F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_queries, frame_num, frame_num)
        frame_dists_query = 1 - frame_sim_query

        if hasattr(self.args.TRAIN, "BALANCE_COEFFICIENT") and self.args.TRAIN.BALANCE_COEFFICIENT:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        else:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        
        cum_dists_regular = dists
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits, "loss_temporal_regular": loss_temporal_regular}
        return return_dict


@HEAD_REGISTRY.register()
class CNN_HyRSM_plusplus_semi(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_plusplus_semi, self).__init__(cfg)
        args = cfg
        # args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50" or self.args.VIDEO.HEAD.BACKBONE_NAME == "inception_v3" :
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        elif hasattr(self.args.TRAIN,"NO_POSITION"):
            self.pe = nn.Sequential()
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)
        

        max_seq_len = self.args.DATA.NUM_INPUT_FRAMES
        temproal_regular = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular[i, j] = 1./(pow(i-j,2)+1.0)
                else:
                    temproal_regular[i, j] = 1.-torch.exp(torch.tensor(-pow(abs(i-j)-self.args.TRAIN.WINDOW_SIZE,2)/self.args.TRAIN.TEMPORAL_BALANCE))
        self.temproal_regular = temproal_regular.cuda()

        temproal_regular_label = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular_label[i, j] = 1.0
            
        self.temproal_regular_label = temproal_regular_label.cuda()

    def get_feats(self, support_images, target_images, support_labels, support_images_unlabel=None, use_unlabel=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        if use_unlabel:
            support_unlabel_features = self.backbone(support_images_unlabel).squeeze()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES
        if use_unlabel:
            Support_unlabel_num = support_unlabel_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]
        if use_unlabel:
            support_unlabel_features = self.relu(self.temporal_atte_before(self.pe(support_unlabel_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY
        

        # pseudo labeling for unlabeled data
        if use_unlabel:
            class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

            support_features_ext = class_prototypes.unsqueeze(0).repeat(Support_unlabel_num,1,1,1)
            support_unlabel_features_ext = support_unlabel_features.unsqueeze(1)
            
            feature_in = torch.cat([support_features_ext.mean(2), support_unlabel_features_ext.mean(2)], 1)
            # feature_in = self.temporal_atte(feature_in, feature_in, feature_in)  # .view(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)) [35, 6, 2048]  45%
            feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)).detach() 
            support_features_a = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
            support_features_a= self.layer2(support_features_a.permute(0,2,1)).permute(0,2,1).reshape(Support_unlabel_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_unlabel_features_a = self.layer2(torch.cat([support_unlabel_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)
            # unique_labels = torch.unique(support_labels)
            n_queries = support_unlabel_features_a.shape[0]
            n_support = support_features_a.shape[1]
            frame_num = support_features_a.shape[2]
            dim = support_features_a.shape[-1]
            support_features_a = rearrange(support_features_a, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
            frame_sim = torch.matmul(F.normalize(support_features_a, dim=2), F.normalize(support_unlabel_features_a, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
            frame_dists = 1 - frame_sim
            dists = frame_dists
            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
            class_dists = - torch.stack(class_dists).detach()   # [5, 35]
            class_dists = rearrange(class_dists, 'c q -> q c')
            pseudo_label_class = torch.softmax(class_dists/self.args.TRAIN.SEMI_TEMPORAL, dim=-1)
            max_probs_class, targets_u_class = torch.max(pseudo_label_class,dim=-1)
            mask_class = max_probs_class.ge(self.args.TRAIN.SEMI_THRESHOLD).float()
            index_class = torch.where(mask_class==1.)
            if torch.any(index_class[0].bool()):
                targets_u_class_refine = targets_u_class[index_class].float()
                support_unlabel_features_refine = support_unlabel_features[index_class]
                support_features = torch.cat([support_features, support_unlabel_features_refine], dim=0)
                support_labels = torch.cat([support_labels, targets_u_class_refine], dim=0)



        # update prototype
        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        if 'target_set_weakly' in inputs:
            support_images_unlabel = inputs["target_set_weakly"]
            use_unlabel=True
        else:
            support_images_unlabel = None
            use_unlabel = False
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels, support_images_unlabel, use_unlabel)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)
        dim = support_features.shape[-1]

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        frame_sim_support = torch.matmul(F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_support*n_queries, frame_num, frame_num)
        frame_dists_support = 1 - frame_sim_support

        frame_sim_query = torch.matmul(F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_queries, frame_num, frame_num)
        frame_dists_query = 1 - frame_sim_query

        if hasattr(self.args.TRAIN, "BALANCE_COEFFICIENT") and self.args.TRAIN.BALANCE_COEFFICIENT:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        else:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        
        cum_dists_regular = dists
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits, "loss_temporal_regular": loss_temporal_regular}
        return return_dict




@HEAD_REGISTRY.register()
class CNN_BiMHM_MoLo(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_BiMHM_MoLo, self).__init__(cfg)
        args = cfg
        self.args = cfg
        last_layer_idx = -1
        self.backbone = nn.Sequential(*list(self.backbone.children())[:last_layer_idx])
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            if hasattr(self.args.TRAIN,"TEMP_COFF") and self.args.TRAIN.TEMP_COFF:
                self.scale = self.args.TRAIN.TEMP_COFF
                self.scale_motion = self.args.TRAIN.TEMP_COFF
            else:
                self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale.data.fill_(1.0)

                self.scale_motion = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale_motion.data.fill_(1.0)

        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
            # self.mid_dim = 256
            self.pre_reduce = nn.Sequential()
            # self.pre_reduce = nn.Conv2d(2048, 256, kernel_size=1, padding=0, groups=4)
        else:
            self.mid_dim = 512
            self.pre_reduce = nn.Sequential()
            # self.pre_reduce = nn.Conv2d(512, 512, kernel_size=1, padding=0)  # nn.Sequential()
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.class_token_motion = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head_k = self.mid_dim//self.args.TRAIN.HEAD, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head_k = self.mid_dim//self.args.TRAIN.HEAD, dropout_atte = 0.2)
            
        else:
            
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.factor = 8
        self.motion_reduce = nn.Conv3d(self.mid_dim, self.mid_dim//self.factor, kernel_size=(3,3,3), padding=(1,1,1), groups=1)
        self.motion_conv = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim//self.factor, kernel_size=3, padding=1, groups=1)
        self.motion_up = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim, kernel_size=1, padding=0, groups=1)
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        
        bilinear = True
        # factor = 2 if bilinear else 1
        factor = 1
        n_classes = 3
        self.up1 = Up2(self.mid_dim//self.factor, 128 // factor, bilinear, kernel_size=2)
        self.up2 = Up2(128, 32 // factor, bilinear, kernel_size=4)
        self.up3 = Up2(32, 16, bilinear, kernel_size=4)
        self.outc = OutConv(16, n_classes)
        # set_trace()
        
    

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.pre_reduce(self.backbone(support_images)).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.pre_reduce(self.backbone(target_images)).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        support_features_motion = self.motion_reduce(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)   # [40, 128, 7, 7]
        target_features_motion = self.motion_reduce(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)
        support_features_motion_conv = self.motion_conv(support_features_motion)   # [40, 128, 7, 7]
        target_features_motion_conv = self.motion_conv(target_features_motion)
        support_features_motion = support_features_motion_conv.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,1:] - support_features_motion.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,:-1]
        support_features_motion = support_features_motion.reshape(-1, dim//self.factor, 7, 7)
        # support_features_motion = self.relu(self.motion_up(support_features_motion))

        target_features_motion = target_features_motion_conv.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,1:] - target_features_motion.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,:-1]
        target_features_motion = target_features_motion.reshape(-1, dim//self.factor, 7, 7)
        # target_features_motion = self.relu(self.motion_up(target_features_motion))
        feature_motion_recons = torch.cat([support_features_motion, target_features_motion], dim=0)
        feature_motion_recons = self.up1(feature_motion_recons)
        feature_motion_recons = self.up2(feature_motion_recons)
        feature_motion_recons = self.up3(feature_motion_recons)
        
        feature_motion_recons = self.outc(feature_motion_recons)
        support_features_motion = self.relu(self.motion_up(support_features_motion))
        target_features_motion = self.relu(self.motion_up(target_features_motion))
        support_features_motion = self.avg_pool(support_features_motion).squeeze().reshape(-1, self.args.DATA.NUM_INPUT_FRAMES-1, dim)
        target_features_motion = self.avg_pool(target_features_motion).squeeze().reshape(-1, self.args.DATA.NUM_INPUT_FRAMES-1, dim)
        support_bs = int(support_features_motion.shape[0])
        target_bs = int(target_features_motion.shape[0])
        support_features_motion = torch.cat((self.class_token_motion.expand(support_bs, -1, -1), support_features_motion), dim=1)
        target_features_motion = torch.cat((self.class_token_motion.expand(target_bs, -1, -1), target_features_motion), dim=1)
        target_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(target_features_motion)))   # [5, 9, 2048]
        support_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(support_features_motion)))

        


        support_features = self.avg_pool(support_features).squeeze()
        target_features = self.avg_pool(target_features).squeeze()

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES
        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        # support_features = self.temporal_atte(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)).unsqueeze(0).repeat(Query_num,1,1,1)   # [35, 5, 8, 2048]   V0
        # target_features = self.temporal_atte(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))# .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]
        support_bs = int(support_features.shape[0])
        target_bs = int(target_features.shape[0])
        support_features = torch.cat((self.class_token.expand(support_bs, -1, -1), support_features), dim=1)
        target_features = torch.cat((self.class_token.expand(target_bs, -1, -1), target_features), dim=1)
        support_features = self.relu(self.temporal_atte_before(self.pe(support_features)))   # [5, 9, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features)))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                if hasattr(self.args.TRAIN, "USE_LOCAL") and self.args.TRAIN.USE_LOCAL:
                    class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
                else:
                    class_logits = self.classification_layer(torch.cat([support_features.mean(1)+support_features_motion.mean(1), target_features.mean(1)+target_features_motion.mean(1)], 0))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None
        

        unique_labels = torch.unique(support_labels)
        support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_features_motion = [torch.mean(torch.index_select(support_features_motion, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features_motion = torch.stack(support_features_motion)
        
        return support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]
        if self.training and hasattr(self.args.TRAIN, "USE_FLOW"):
            support_images_re = inputs["support_set_flow"].reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)[:,:(self.args.DATA.NUM_INPUT_FRAMES-1),:,:,:]
            target_images_re = inputs["target_set_flow"].reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)[:,:(self.args.DATA.NUM_INPUT_FRAMES-1),:,:,:]
            input_recons = torch.cat([support_images_re, target_images_re], dim=0).reshape(-1, 3, 224, 224)
        else:
            support_images_re = support_images.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)
            target_images_re = target_images.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)
            # support_images, support_labels, target_images = inputs
            input_recons = torch.cat([support_images_re[:,1:,:]-support_images_re[:,:-1,:], target_images_re[:,1:,:]- target_images_re[:,:-1,:]], dim=0).reshape(-1, 3, 224, 224)

        support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons = self.get_feats(support_images, target_images, support_labels)
        
        # 
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        # global
        support_features_g = support_features[:,0,:]
        target_features_g = target_features[:,0,:]
        support_features = support_features[:,1:,:]
        target_features = target_features[:,1:,:]

        # support to query
        class_sim_s2q = cos_sim(support_features, target_features_g)  # [5, 8, 35]
        class_dists_s2q = 1 - class_sim_s2q
        class_dists_s2q = [torch.sum(torch.index_select(class_dists_s2q, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q = torch.stack(class_dists_s2q).squeeze(1)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_s2q = rearrange(class_dists_s2q * self.scale, 'c q -> q c')

        # query to support 
        class_sim_q2s = cos_sim(target_features, support_features_g)  # [35, 8, 5]
        class_dists_q2s = 1 - class_sim_q2s   
        class_dists_q2s = [torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze(2)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')

        # global
        support_features_motion_g = support_features_motion[:,0,:]
        target_features_motion_g = target_features_motion[:,0,:]
        support_features_motion = support_features_motion[:,1:,:]
        target_features_motion = target_features_motion[:,1:,:]

        # support to query
        class_sim_s2q_motion = cos_sim(support_features_motion, target_features_motion_g)  # [5, 8, 35]
        class_dists_s2q_motion = 1 - class_sim_s2q_motion
        class_dists_s2q_motion = [torch.sum(torch.index_select(class_dists_s2q_motion, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q_motion = torch.stack(class_dists_s2q_motion).squeeze(1)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_s2q_motion = rearrange(class_dists_s2q_motion * self.scale_motion, 'c q -> q c')

        # query to support 
        class_sim_q2s_motion = cos_sim(target_features_motion, support_features_motion_g)  # [35, 8, 5]
        class_dists_q2s_motion = 1 - class_sim_q2s_motion   
        class_dists_q2s_motion = [torch.sum(torch.index_select(class_dists_q2s_motion, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s_motion = torch.stack(class_dists_q2s_motion).squeeze(2)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_q2s_motion = rearrange(class_dists_q2s_motion * self.scale_motion, 'c q -> q c')

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query
        if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
            cum_dists = dists.min(3)[0].sum(2)
        else:
            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
            

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')

        support_features_motion = rearrange(support_features_motion, 'b s d -> (b s) d')  # [200, 2048]
        target_features_motion = rearrange(target_features_motion, 'b s d -> (b s) d')    # [200, 2048]
        frame_sim_motion = cos_sim(target_features_motion, support_features_motion)    # [200, 200]
        frame_dists_motion = 1 - frame_sim_motion   
        dists_motion = rearrange(frame_dists_motion, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query
        if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
            # cum_dists_motion = OTAM_cum_dist(dists_motion)
            cum_dists_motion = dists_motion.min(3)[0].sum(2)
        else:
            cum_dists_motion = dists_motion.min(3)[0].sum(2) + dists_motion.min(2)[0].sum(2)
        class_dists_motion = [torch.mean(torch.index_select(cum_dists_motion, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_motion = torch.stack(class_dists_motion)
        class_dists_motion = rearrange(class_dists_motion, 'c q -> q c')
        
        if hasattr(self.args.TRAIN, "LOGIT_BALANCE_COFF") and self.args.TRAIN.LOGIT_BALANCE_COFF:
            class_dists = class_dists + self.args.TRAIN.LOGIT_BALANCE_COFF*class_dists_motion
        else:
            class_dists = class_dists + 0.3*class_dists_motion
        
        if self.training:
            loss_recons = (feature_motion_recons - input_recons) ** 2   # [280, 3, 224, 224]
            loss_recons = loss_recons.mean()  # [N, L], mean loss per patch
        else:
            loss_recons = torch.tensor(0.)

        
        return_dict = {'logits': - class_dists , 'class_logits': class_logits, "logits_s2q": -class_dists_s2q, "logits_q2s": -class_dists_q2s, "logits_s2q_motion": -class_dists_s2q_motion, "logits_q2s_motion": -class_dists_q2s_motion, "loss_recons": loss_recons,}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


@HEAD_REGISTRY.register()
class CNN_OTAM_CLIPFSAR(CNN_FSHead):
    def __init__(self, cfg):
        super(CNN_OTAM_CLIPFSAR, self).__init__(cfg)
        self.args = cfg
        if cfg.VIDEO.HEAD.BACKBONE_NAME=="RN50":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual    # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 1024
        elif cfg.VIDEO.HEAD.BACKBONE_NAME=="ViT-B/16":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual   # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 512
        with torch.no_grad():
            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                text_templete = [self.args.TEST.PROMPT.format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
            text_templete = tokenize(text_templete).cuda()
            self.text_features_train = backbone.encode_text(text_templete)

            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                text_templete = [self.args.TEST.PROMPT.format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
            else:
                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
            text_templete = tokenize(text_templete).cuda()
            
            self.text_features_test = backbone.encode_text(text_templete)
        
        self.mid_layer = nn.Sequential()
        self.classification_layer = nn.Sequential() 
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)
        
        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2, depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)


    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze()
            target_features = self.backbone(target_images).squeeze()

            dim = int(support_features.shape[1])

            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features_text = None
            
        else:
            support_features = self.backbone(support_images).squeeze()
            target_features = self.backbone(target_images).squeeze()
            dim = int(target_features.shape[1])
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            # support_real_class = torch.unique(support_real_class)
            support_features_text = self.text_features_test[support_real_class.long()]

        return support_features, target_features, support_features_text

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels'] # [200, 3, 224, 224] inputs["real_support_labels"]
        unique_labels = torch.unique(support_labels)
        if self.training:
            support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
            support_bs = support_features.shape[0]
            target_bs = target_features.shape[0]

            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale
            else:
                class_text_logits = None
            
            if self.training:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
            else:
                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            
            target_features = self.context2(target_features, target_features, target_features)
            context_support = self.mid_layer(context_support) 
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
            support_features = torch.cat([support_features, context_support], dim=1)
            support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')  
            target_features = rearrange(target_features, 'b s d -> (b s) d')    

            frame_sim = cos_sim(target_features, support_features)  
            frame_dists = 1 - frame_sim
            
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

            # calculate query -> support and support -> query
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        
        else:
            if hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT:
                support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class)
                text_features = [torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                text_features = torch.stack(text_features)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)
                
                cum_dists = -logits_per_image # 
                class_text_logits = None

                
            elif hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE:
                # text_features = self.text_features_test[support_real_class.long()]
                support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class)
                text_features = [torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                text_features = torch.stack(text_features)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)
                
                class_text_logits = None

                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
                
                target_features = self.context2(target_features, target_features, target_features)
                context_support = self.mid_layer(context_support)  # F.relu(self.mid_layer(context_support))
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    pass
                else:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')  
                target_features = rearrange(target_features, 'b s d -> (b s) d')    

                frame_sim = cos_sim(target_features, support_features)    
                frame_dists = 1 - frame_sim
                
                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists_visual = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists_visual = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
                cum_dists_visual_soft = F.softmax((8-cum_dists_visual)/8., dim=1)
                if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF:
                    cum_dists = -(logits_per_image.pow(self.args.TRAIN.TEXT_COFF)*cum_dists_visual_soft.pow(1.0-self.args.TRAIN.TEXT_COFF))
                else:
                    cum_dists = -(logits_per_image.pow(0.9)*cum_dists_visual_soft.pow(0.1))
                
                class_text_logits = None

            else:
                support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

                
                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
                
                target_features = self.context2(target_features, target_features, target_features)
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    pass
                else:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
                target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

                frame_sim = cos_sim(target_features, support_features)    # [200, 200]
                frame_dists = 1 - frame_sim
                
                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, "class_logits": class_text_logits}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


@HEAD_REGISTRY.register()
class CLIPFSAR(CNN_FSHead):
    def __init__(self, cfg):
        super(CLIPFSAR, self).__init__(cfg)
        self.args = cfg
        if self.args.VIDEO.HEAD.BACKBONE_NAME=="RN50":
            backbone, self.preprocess = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=self.args, jit=False)   # ViT-B/16
            self.backbone = backbone # backbone.visual
            self.mid_dim = 1024
        elif self.args.VIDEO.HEAD.BACKBONE_NAME=="ViT-B/16":
            backbone, self.preprocess = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=self.args, jit=False)   # ViT-B/16
            self.backbone = backbone # backbone.visual
            self.mid_dim = 512

        if self.args.TRAIN.DATASET == 'FSARVideoDataset':
            print('use FSARVideoDataset')
            self.class_name_unique = self.args.class_name_unique
            self.class_name = classname2words(self.class_name_unique)

            with torch.no_grad():
                print('get_text_feat for class names:\n', self.class_name)
                self.class_name_feat = self.get_text_feat(self.class_name, _cuda=True)
            self.trainset_textfeat = self.class_name_feat[:self.args.NUM_CLASS]  # [n, text_dim] n is class num in train set
            # {'Shotput':feat1, 'PoleVault':feat2, ...}
            self.cls_feat_dic = {self.class_name_unique[i]: self.class_name_feat[i] for i in range(len(self.class_name_unique))}
        elif self.args.TRAIN.DATASET == 'Ssv2_few_shot':
            print('use Ssv2_few_shot')
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME

            with torch.no_grad():
                if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                    text_templete = [self.args.TEST.PROMPT.format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                else:
                    text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
                text_templete = tokenize(text_templete).cuda()
                self.trainset_textfeat = self.backbone.encode_text(text_templete)

                if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                    text_templete = [self.args.TEST.PROMPT.format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                else:
                    text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
                text_templete = tokenize(text_templete).cuda()
                self.text_features_test = self.backbone.encode_text(text_templete)

        # del self.backbone.transformer

        self.mid_layer = nn.Sequential()
        self.classification_layer = nn.Sequential()
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)

        self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2, depth=1)

    def get_text_feat(self, text_label, temp=None, _cuda=True):
        '''
        Args:
            text_label: list ['boxing punching bag', 'kayaking', 'throw discus', ... ]
        Returns:
            text_features: torch tensor [N, dim]
        '''
        prompt_texts = hard_prompt(text_label, temp=temp)
        text = tokenize(prompt_texts)   # [N x K, 77]
        if _cuda:
            text = text.to('cuda')
        try:
            text_features = self.backbone.module.encode_text(text)   # [N x K, 1024]  unordered
        except:
            text_features = self.backbone.encode_text(text)
        return text_features

    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        support_features = self.backbone.encode_image(support_images).squeeze()
        target_features = self.backbone.encode_image(target_images).squeeze()
        dim = int(support_features.shape[1])
        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        return support_features, target_features

    def forward(self, task_dict):
        support_images, support_labels, target_images, target_labels = task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'], task_dict["target_labels"]
        unique_labels = torch.unique(support_labels)

        support_features, target_features = self.get_feats(support_images, target_images)

        if self.args.TRAIN.DATASET == 'FSARVideoDataset':
            support_class_name, target_class_name = task_dict['support_class_name'], task_dict['target_class_name']
            support_text_feat = torch.stack([self.cls_feat_dic.get(item[0]) for item in support_class_name]).cuda()  # [N x K, text_dim]
            target_text_feat = torch.stack([self.cls_feat_dic.get(item[0]) for item in target_class_name]).cuda()  # [N x M, text_dim]
        elif self.args.TRAIN.DATASET == 'Ssv2_few_shot':
            support_real_class, target_real_class = task_dict['real_support_labels'], task_dict['real_target_labels']
            if self.training:
                support_text_feat = self.trainset_textfeat[support_real_class.long()]
                target_text_feat = self.trainset_textfeat[target_real_class.long()]
            else:
                support_text_feat = self.text_features_test[support_real_class.long()]
                target_text_feat = self.text_features_test[target_real_class.long()]
        else:
            raise NotImplementedError

        if self.training:
            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE") and self.args.TRAIN.USE_CLASSIFICATION_VALUE > 0:
                feature_classification_in = torch.cat([support_features, target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.trainset_textfeat) * self.scale
            else:
                class_text_logits = None

            context_support = support_text_feat.unsqueeze(1)

            target_features = self.context2(target_features, target_features, target_features)
            context_support = self.mid_layer(context_support)
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
            support_features = torch.cat([support_features, context_support], dim=1)
            support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.MERGE_BEFORE:
                pass
            else:
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]

            support_features = rearrange(support_features, 'b s d -> (b s) d')
            target_features = rearrange(target_features, 'b s d -> (b s) d')

            frame_sim = cos_sim(target_features, support_features)
            frame_dists = 1 - frame_sim

            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

            # calculate query -> support and support -> query
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        else:
            text_prototypes = [
                torch.mean(torch.index_select(support_text_feat, 0, extract_class_indices(support_labels, c)), dim=0)
                for c in unique_labels]
            text_prototypes = torch.stack(text_prototypes)  # [5, text_dim]

            if hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.EVAL_TEXT:
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_prototypes / text_prototypes.norm(dim=1, keepdim=True)
                # cosine similarity as logits
                logit_scale = self.scale  # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)

                cum_dists = -logits_per_image  #
                class_text_logits = None

            elif hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE:
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_prototypes / text_prototypes.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale  # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)

                class_text_logits = None

                feature_classification_in = torch.cat([support_features, target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.trainset_textfeat) * self.scale

                context_support = support_text_feat.unsqueeze(1)

                target_features = self.context2(target_features, target_features, target_features)
                context_support = self.mid_layer(context_support)  # F.relu(self.mid_layer(context_support))
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.MERGE_BEFORE:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.MERGE_BEFORE:
                    pass
                else:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')
                target_features = rearrange(target_features, 'b s d -> (b s) d')

                frame_sim = cos_sim(target_features, support_features)
                frame_dists = 1 - frame_sim

                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists_visual = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists_visual = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
                cum_dists_visual_soft = F.softmax((8-cum_dists_visual)/8., dim=1)
                if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TEXT_COFF:
                    cum_dists = -(logits_per_image.pow(self.args.TEXT_COFF)*cum_dists_visual_soft.pow(1.0-self.args.TEXT_COFF))
                else:
                    cum_dists = -(logits_per_image.pow(0.9) * cum_dists_visual_soft.pow(0.1))

                class_text_logits = None

            else:
                feature_classification_in = torch.cat([support_features, target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.trainset_textfeat) * self.scale

                context_support = support_text_feat.unsqueeze(1)

                target_features = self.context2(target_features, target_features, target_features)
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.MERGE_BEFORE:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.MERGE_BEFORE:
                    pass
                else:
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]

                support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
                target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200, 2048]

                frame_sim = cos_sim(target_features, support_features)  # [200, 200]
                frame_dists = 1 - frame_sim

                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, "class_logits": class_text_logits}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


def freeze_parameters(module):
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = False
    elif isinstance(module, (list, tuple)):
        for item in module:
            freeze_parameters(item)
    elif isinstance(module, dict):
        for key, item in module.items():
            freeze_parameters(item)

class Transformer_Block(nn.Module):
    def __init__(self, heads=8, dim=2048,dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim = dim, num_heads = heads, dropout = dropout_atte)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout_ffn)

    def forward(self, q, k, v, attn_mask = None):
        q , k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2) # NLD -> LND
        if attn_mask == None:
            x = self.attn(self.ln1(q), self.ln1(k), self.ln1(v), need_weights=False)[0] + q #residual
        else:
            x = self.attn(self.ln1(q), self.ln1(k), self.ln1(v), need_weights=False, attn_mask = attn_mask)[0] + q
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ff(self.ln2(x)) + x
        return x

@HEAD_REGISTRY.register()
class LGA(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(LGA, self).__init__(cfg)

        self.args = cfg
        if cfg.VIDEO.HEAD.BACKBONE_NAME=="RN50":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual    # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 1024
        elif cfg.VIDEO.HEAD.BACKBONE_NAME=="ViT-B/16":
            backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)   # ViT-B/16
            self.backbone = backbone.visual   # model.load_state_dict(state_dict)
            self.backbone = nn.DataParallel(self.backbone, device_ids=[ i  for i in range(cfg.NUM_GPUS)]) #Backbone
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            self.mid_dim = 512

        if hasattr(self.args.TRAIN, "FROZEN_BACKBONE") and self.args.TRAIN.FROZEN_BACKBONE:
            print("========FROZEN_BACKBONE=========")
            freeze_parameters(self.backbone)


        with torch.no_grad():
            text_train = cfg.TRAIN.SUBACTION_TXT_TRAIN #len(text_templete) = 64
            train_token=[]
            for item in text_train:
                action_label = item["Action Label"]
                scene_description = item["Scene Description"].split(',')
                scene_description += [" "] * (8 - len(scene_description))
                sub_action = item["Sub-Action Description"]
                action_description = [action_label]+scene_description+sub_action
                train_token.append(action_description)
            # text_train_sub_action
            flat_train_token = [item for sublist in train_token for item in sublist]
            text_train_sub_action = tokenize(flat_train_token).cuda()
            self.text_features_train = backbone.encode_text(text_train_sub_action).view(-1,12,self.mid_dim) #[64,12,512]

            text_test = cfg.TRAIN.SUBACTION_TXT_TEST #len(text_templete) = 24
            test_token=[]
            for item in text_test:
                action_label = item["Action Label"]
                scene_description = item["Scene Description"].split(',')
                scene_description += [" "] * (8 - len(scene_description))
                sub_action = item["Sub-Action Description"]
                action_description = [action_label]+scene_description+sub_action
                test_token.append(action_description)

            flat_test_token = [item for sublist in test_token for item in sublist]
            text_test_sub_action = tokenize(flat_test_token).cuda()
            self.text_features_test = backbone.encode_text(text_test_sub_action).view(-1,12,self.mid_dim) #[24,12,512]

            # None_token = [" "]
            # text_None_sub_action = tokenize(None_token).cuda()
            # self.text_features_None = backbone.encode_text(text_None_sub_action)


        self.mid_layer = nn.Sequential()
        self.classification_layer = nn.Sequential()

        self.global_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))

        # ===========Attn_mask 2: 固定的Attn_mask==========
        support_weights = torch.tensor([[1,1,1,1,0,0,0,0,1,0,0],
                        [0,0,1,1,1,1,0,0,0,1,0],
                        [0,0,0,0,1,1,1,1,0,0,1]]
                        ,dtype=torch.float32)
        support_attn_masks = []
        for weight in support_weights:
            attn_mask = torch.ger(weight, weight) # outer product to get attn_mask
            support_attn_masks.append(attn_mask)

        self.support_attn = nn.Parameter(torch.stack(support_attn_masks, dim=0).cuda(), requires_grad=False) #[3,11,11]

        target_weights = torch.tensor([[1,1,1,1,0,0,0,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,0,0,1,1,1,1]]
                        ,dtype=torch.float32)
        target_attn_masks = []
        for weight in target_weights:
            attn_mask = torch.ger(weight, weight)
            target_attn_masks.append(attn_mask)
        self.target_attn = nn.Parameter(torch.stack(target_attn_masks, dim=0).cuda(), requires_grad=False)


        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)


        self.heads = 8
        self.fine_attn = Transformer_Block(dim=self.mid_dim, heads = self.heads, dropout_atte = 0.2)

    # Feature extraction
    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze() #[5*8,1024]
            # print(support_features.shape)
            if self.args.NUM_GPUS > 1 and self.args.SINGLE_PROCESS:
                support_features = torch.cat([support_features],dim=0).cuda(0)

            # os.system("nvidia-smi")
            target_features = self.backbone(target_images).squeeze() #[15*8,1024]
            if self.args.NUM_GPUS > 1 and self.args.SINGLE_PROCESS:
                target_features = torch.cat([target_features],dim=0).cuda(0)
            # os.system("nvidia-smi")

            dim = int(support_features.shape[-1])

            # print(support_features.shape)
            # support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 197, dim)  #[5*8,1024]
            # target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 197, dim) #[15*8,1024]
            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 1, dim)  #[5*8,1024]
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 1, dim) #[15*8,1024]
            support_features_text = None
            support_features = support_features[:,:,0,:] # [5,8,1024]
            target_features = target_features[:,:,0,:]


        else:
            support_features = self.backbone(support_images).squeeze()
            if self.args.NUM_GPUS > 1 and self.args.SINGLE_PROCESS:
                support_features = torch.cat([support_features],dim=0).cuda(0)
            target_features = self.backbone(target_images).squeeze()
            if self.args.NUM_GPUS > 1 and self.args.SINGLE_PROCESS:
                support_features = torch.cat([support_features],dim=0).cuda(0)
            dim = int(target_features.shape[-1])
            # support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 197, dim)  #[5*8,1024]
            # target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 197, dim) #[15*8,1024]
            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 1, dim)  #[5*8,1024]
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, 1, dim) #[15*8,1024]
            # support_real_class = torch.unique(support_real_class)
            # support_features_text = self.text_features_test[support_real_class.long()]
            support_features_text = None

            support_features = support_features[:,:,0,:] # [5,8,1024]
            target_features = target_features[:,:,0,:]


        return support_features, target_features, support_features_text

    # Visual Feature Cluster
    def culster(self, source_features, target_len=3):
        """
        input:
        features: [batch, len, dim]
        target_len: culster to target_len

        output:
        features: [batch, target_len, dim]
        current_indices: list # segment result
        """
        features = source_features
        batch_size = features.size(0)

        current_indices = [list([i] for i in range(self.args.DATA.NUM_INPUT_FRAMES)) for _ in range(batch_size)]
        while features.size(1) > target_len:

            norm_features = F.normalize(features, p=2, dim=-1)  #
            similarities = F.cosine_similarity(norm_features[:, :-1], norm_features[:, 1:], dim=-1)  # [batch, 7]
            index = torch.argmax(similarities, dim=-1)  # [batch]

            new_features = []
            for b in range(batch_size):
                idx = index[b].item()
                wg1 = len(current_indices[b][idx])
                wg2 = len(current_indices[b][idx+1])
                fused_feature = (wg1*features[b, idx] + wg2*features[b, idx + 1]) / (wg1+wg2)

                new_feature = torch.cat([features[b, :idx], fused_feature.unsqueeze(0), features[b, idx + 2:]], dim=0)
                new_features.append(new_feature)

                # Cluster list update
                current_indices[b][idx] = current_indices[b][idx]+current_indices[b][idx+1]
                current_indices[b].pop(idx+1)

            features = torch.stack(new_features)  # [batch, len, dim]

        # Finished current_indices
        # generate current_indices_new & feature_new
        current_indices_final = []
        feature_final = []
        for i in range(batch_size):
            current_indices_new = []
            for j in range(target_len):
                # add overlap
                if j == 0:
                    current_indices_new.append(current_indices[i][j] + [current_indices[i][j+1][0]])
                    feature_new = source_features[i,current_indices_new[j],:]
                elif j == target_len-1:
                    current_indices_new.append([current_indices[i][j-1][-1]] + current_indices[i][j])
                    feature_new = torch.cat([feature_new, source_features[i,current_indices_new[j],:]],dim=0)
                else:
                    current_indices_new.append([current_indices[i][j-1][-1]] + current_indices[i][j] + [current_indices[i][j+1][0]])
                    feature_new = torch.cat([feature_new, source_features[i,current_indices_new[j],:]],dim=0)



            feature_final.append(feature_new)
            current_indices_new = [1]*len(current_indices_new[0])+[2]*len(current_indices_new[1])+[3]*len(current_indices_new[2])
            current_indices_final.append(current_indices_new)
        feature_final = torch.stack(feature_final)
        current_indices_final = torch.tensor(current_indices_final)

        return feature_final, current_indices_final

    def video_video_classify_head(self, support_features, target_features, support_labels, support_weights=None, target_weights=None):
        # suppoprt_features: [15, 8, 512]
        # target_features: [10, 8, 512]
        # support_weights: [15, 12]
        # target_weights: [10, 12]

        unique_labels = torch.unique(support_labels)

        #  weights initialization
        if support_weights == None:
            support_weights = torch.tensor([1,1,1,1,2,2,2,2,3,3,3,3]).unsqueeze(0).expand(support_features.shape[0], -1).cuda()
        if target_weights == None:
            target_weights = torch.tensor([1,1,1,1,2,2,2,2,3,3,3,3]).unsqueeze(0).expand(target_features.shape[0], -1).cuda()

        target_weights = target_weights.unsqueeze(1).expand(-1, support_weights.shape[0], -1) #[target_num, support_num, 12]
        support_weights = support_weights.unsqueeze(0).expand(target_weights.shape[0], -1, -1) #[target_num, support_num, 12]

        weights = target_weights.unsqueeze(2)*support_weights.unsqueeze(-1)
        weights = ((weights == 1) | (weights == 4) | (weights == 9)).float()
        weights[weights !=1] = 1e6

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)
        frame_dists = 1 - frame_sim

        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [q*num, s, 8, 8]

        if hasattr(self.args.TRAIN, "VIDEO_MATCH") and self.args.TRAIN.VIDEO_MATCH == "BI-MHM":
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = dists.min(3)[0].sum(2)
            else:
                cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)  # [5*query_per_class,5]

        elif hasattr(self.args.TRAIN, "VIDEO_MATCH") and self.args.TRAIN.VIDEO_MATCH == "FINEACTION":
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                # cum_dists = dists[:,:,:4,:4].min(3)[0].sum(2) + dists[:,:,4:8,4:8].min(3)[0].sum(2) + dists[:,:,8:12,8:12].min(3)[0].sum(2)
                cum_dists = (weights*dists).min(3)[0].sum(2)
            else:
                cum_dists = (weights*dists).min(3)[0].sum(2) + (weights*dists).min(2)[0].sum(2)  # [5*query_per_class,5]
                # cum_dists_ = dists[:,:,:4,:4].min(3)[0].sum(2) + dists[:,:,4:8,4:8].min(3)[0].sum(2) + dists[:,:,8:12,8:12].min(3)[0].sum(2) \
                #     + dists[:,:,:4,:4].min(2)[0].sum(2) + dists[:,:,4:8,4:8].min(2)[0].sum(2) + dists[:,:,8:12,8:12].min(2)[0].sum(2)  # [5*query_per_class,5]

        elif hasattr(self.args.TRAIN, "VIDEO_MATCH") and self.args.TRAIN.VIDEO_MATCH == "TOPK":
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = dists.min(3)[0].topk(k=2,dim=2,largest=False)[0].sum(2)
            else:
                cum_dists = dists.min(3)[0].topk(k=8,dim=2,largest=False)[0].sum(2) + dists.min(2)[0].topk(k=8,dim=2,largest=False)[0].sum(2)  # [5*query_per_class,5]

        elif hasattr(self.args.TRAIN, "VIDEO_MATCH") and self.args.TRAIN.VIDEO_MATCH == "OTAM":
            # calculate query -> support and support -> query  Frame Classification
            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        # cum_dists [query_num, support_num] [10,15]
        cum_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        cum_dists = torch.stack(cum_dists)
        cum_dists = rearrange(cum_dists, 'c q -> q c')

        return cum_dists


    def forward(self, inputs):
        """
        query_logits: [query_num, 5]
        """
        support_images, support_labels, target_images  = inputs['support_set'], inputs['support_labels'], inputs['target_set']  # [200, 3, 224, 224] inputs["real_support_labels"]
        support_real_class, target_real_class = inputs['real_support_labels'], inputs['real_target_labels']
        # support_real_class
        # set_trace()

        # Video featrue extraction
        support_video_features, target_video_features, _ = self.get_feats(support_images, target_images, support_labels) #[5*8,1024],[15*8,1024]
        support_bs = support_video_features.shape[0]
        target_bs = target_video_features.shape[0]

        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            feature_classification_in = torch.cat([support_video_features,target_video_features], dim=0)
            feature_classification = self.classification_layer(feature_classification_in).mean(1)
            class_text_logits = cos_sim(feature_classification, self.text_features_train[:,0,:])*self.scale
        else:
            class_text_logits = None

        # Text feature extraction
        if self.training:
            # 这里text_features_train [64,1024]
            context_support = self.text_features_train[support_real_class.long()] # [5, 4, dim]
            context_support_global = context_support[:, 0, :].unsqueeze(1)
            context_support_fine =  context_support[:, 9:, :] # [5, 3, dim]
            if hasattr(self.args.TRAIN, "REAPEAT_LABEL") and self.args.TRAIN.REAPEAT_LABEL:
                context_support_fine = context_support_global.repeat(1,3,1)
            context_support_element = context_support[:, 1:9, :] # [5, 8, dim]

            # context_target = self.text_features_train[target_real_class.long()] # [25, 4, dim]
            # context_target_global = context_target[:, 0, :].unsqueeze(1)
            # context_target_fine =  context_target[:, 9:, :] # [25, 3, dim]
            # context_target_element = context_target[:, 1:9, :] # [25, 8, dim]

        else:
            context_support = self.text_features_test[support_real_class.long()] # [5, 4, dim]
            context_support_global = context_support[:, 0, :].unsqueeze(1)
            context_support_fine =  context_support[:, 9:, :]
            if hasattr(self.args.TRAIN, "REAPEAT_LABEL") and self.args.TRAIN.REAPEAT_LABEL:
                context_support_fine = context_support_global.repeat(1,3,1)
            context_support_element = context_support[:, 1:9, :]

            #---------up_limit if know query class-------
            # context_target = self.text_features_test[target_real_class.long()] # [25, 4, dim]
            # context_target_global = context_target[:, 0, :].unsqueeze(1)
            # context_target_fine =  context_target[:, 9:, :]
            # context_target_element = context_target[:, 1:9, :]

        unique_labels = torch.unique(support_labels)
        text_features_fine = context_support_fine # [5, 3, 512] or [5, 1, 512]
        text_features_fine = [torch.mean(torch.index_select(text_features_fine, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        text_features_fine = torch.stack(text_features_fine) # [5,3,512]

        text_features_global = context_support_global
        text_features_global = [torch.mean(torch.index_select(text_features_global, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        text_features_global = torch.stack(text_features_global) # [5,3,512]

        if hasattr(self.args.TRAIN, "TEMPORAL_SEGMENT") and self.args.TRAIN.TEMPORAL_SEGMENT == "HARD":
            # Fine_Attention (1)- MASK_ATTEN
            support_weights = torch.tensor([1,1,1,1,2,2,2,2,3,3,3,3]).unsqueeze(0).expand(support_bs, -1).cuda()
            target_weights = torch.tensor([1,1,1,1,2,2,2,2,3,3,3,3]).unsqueeze(0).expand(target_bs, -1).cuda()
            support_weights = nn.Parameter(support_weights, requires_grad=False)
            target_weights = nn.Parameter(target_weights, requires_grad=False)


            support_attn_masks = []
            for weight in support_weights:
                attn_mask = torch.ger(weight, weight)
                support_attn_masks.append(attn_mask)
            support_attn = nn.Parameter(torch.stack(support_attn_masks, dim=0).cuda(), requires_grad=False) # [5, 12, 12]
            support_attn = support_attn.unsqueeze(1).expand(-1, self.heads, -1, -1).reshape(-1, 12, 12) # [N*K*8, 3, 12, 12]
            target_attn_masks = []
            for weight in target_weights:
                attn_mask = torch.ger(weight, weight)
                target_attn_masks.append(attn_mask)
            target_attn = nn.Parameter(torch.stack(target_attn_masks, dim=0).cuda(), requires_grad=False)# [15, 12, 12]
            target_attn = target_attn.unsqueeze(1).expand(-1, self.heads, -1, -1).reshape(-1, 12, 12) # [N*query_cls*8, 3, 12, 12]

            # 控制全局的Atten的结果
            if hasattr(self.args.TRAIN, "Long_Para"):
                long_para = self.args.TRAIN.Long_Para
            else:
                long_para = 0.0
            support_attn_mask_1 =  torch.where(support_attn == 1, torch.tensor(1.0, device=support_attn.device), torch.tensor(long_para, device=support_attn.device))
            support_attn_mask_2 =  torch.where(support_attn == 4, torch.tensor(1.0, device=support_attn.device), torch.tensor(long_para, device=support_attn.device))
            support_attn_mask_3 =  torch.where(support_attn == 9, torch.tensor(1.0, device=support_attn.device), torch.tensor(long_para, device=support_attn.device))

            target_attn_mask_1 =  torch.where(target_attn == 1, torch.tensor(1.0, device=target_attn.device), torch.tensor(long_para, device=target_attn.device))
            target_attn_mask_2 =  torch.where(target_attn == 4, torch.tensor(1.0, device=target_attn.device), torch.tensor(long_para, device=target_attn.device))
            target_attn_mask_3 =  torch.where(target_attn == 9, torch.tensor(1.0, device=target_attn.device), torch.tensor(long_para, device=target_attn.device))

            # self.support_attn_masks = self.support_attn.unsqueeze(0).expand(support_bs*self.heads, -1, -1, -1) # [N*K*8, 3, 8, 8]
            # self.support_attn_masks = self.support_attn_masks.permute(1, 0, 2, 3) # [3, N*K*8, 8, 8]
            # support_features_fine_concat = torch.cat([support_video_features, context_support_fine], dim=1)  # [5, 3+8, 512]
            # support_features_fine_0 = self.fine_attn(support_features_fine_concat, support_features_fine_concat, support_features_fine_concat, attn_mask=self.support_attn_masks[0])[:, :4, :] #[5, 4, 512 ]
            # support_features_fine_1 = self.fine_attn(support_features_fine_concat, support_features_fine_concat, support_features_fine_concat, attn_mask=self.support_attn_masks[1])[:, 2:6, :] #[5, 4, 512 ]
            # support_features_fine_2 = self.fine_attn(support_features_fine_concat, support_features_fine_concat, support_features_fine_concat, attn_mask=self.support_attn_masks[2])[:, 4:8, :] #[5, 4, 512 ]

            self.support_attn_masks = self.target_attn.unsqueeze(0).expand(support_bs*self.heads, -1, -1, -1) # [N*K*8, 3, 8, 8]
            self.support_attn_masks = self.support_attn_masks.permute(1, 0, 2, 3) # [3, N*K*8, 8, 8]

            # for Alignment
            support_segment_0 = support_video_features[:, :4, :]
            support_segment_1 = support_video_features[:, 2:6, :]
            support_segment_2 = support_video_features[:, 4:, :]

            support_features_fine = torch.cat([support_segment_0, support_segment_1, support_segment_2], dim=1)  # [5, 12, 512]

            support_features_fine_textual_0  = support_segment_0 + context_support_fine[:,0,:].unsqueeze(1)
            support_features_fine_textual_1  = support_segment_1 + context_support_fine[:,1,:].unsqueeze(1)
            support_features_fine_textual_2  = support_segment_2 + context_support_fine[:,2,:].unsqueeze(1)

            support_features_fine_textual = torch.cat([support_features_fine_textual_0, support_features_fine_textual_1, support_features_fine_textual_2], dim=1)  # [5, 12, 512]
            if hasattr(self.args.TRAIN, "ATTEN") and self.args.TRAIN.ATTEN == "SELF":
                support_features_fine_textual_0 = torch.cat([support_segment_0, context_support_fine[:,0,:].unsqueeze(1)], dim=1)
                support_features_fine_textual_1 = torch.cat([support_segment_1, context_support_fine[:,1,:].unsqueeze(1)], dim=1)
                support_features_fine_textual_2 = torch.cat([support_segment_2, context_support_fine[:,2,:].unsqueeze(1)], dim=1)

                support_features_fine_0 = self.fine_attn(support_features_fine_textual_0, support_features_fine_textual_0, support_features_fine_textual_0)[:, :-1, :] #[5, 4, 512 ]
                support_features_fine_1 = self.fine_attn(support_features_fine_textual_1, support_features_fine_textual_1, support_features_fine_textual_1)[:, :-1, :] #[5, 4, 512 ]
                support_features_fine_2 = self.fine_attn(support_features_fine_textual_2, support_features_fine_textual_2, support_features_fine_textual_2)[:, :-1, :] #[5, 4, 512 ]
            else:
            #tpye(2)-cross
                support_features_fine_0 = self.fine_attn(support_features_fine_textual, support_features_fine,support_features_fine, attn_mask=support_attn_mask_1)[:, :4, :] #[5, 4, 512 ]
                support_features_fine_1 = self.fine_attn(support_features_fine_textual, support_features_fine,support_features_fine, attn_mask=support_attn_mask_2)[:, 4:8, :] #[5, 4, 512 ]
                support_features_fine_2 = self.fine_attn(support_features_fine_textual, support_features_fine,support_features_fine, attn_mask=support_attn_mask_3)[:, 8:12, :] #[5, 4, 512 ]


            support_features_fine_final = torch.cat([support_features_fine_0, support_features_fine_1, support_features_fine_2], dim=1) #[N way*K shot, 12, 512]

            target_segment_0 = target_video_features[:, :4, :]
            target_segment_1 = target_video_features[:, 2:6, :]
            target_segment_2 = target_video_features[:, 4:, :]

            target_features_fine = torch.cat([target_segment_0, target_segment_1, target_segment_2], dim=1)  # [5, 12, 512]

            target_features_fine_0 = self.fine_attn(target_features_fine, target_features_fine, target_features_fine,attn_mask = target_attn_mask_1)[:, :4, :]  #[5, 4, 512 ]
            target_features_fine_1 = self.fine_attn(target_features_fine, target_features_fine, target_features_fine,attn_mask = target_attn_mask_2)[:, 4:8, :]  #[5, 4, 512 ]
            target_features_fine_2 = self.fine_attn(target_features_fine, target_features_fine, target_features_fine,attn_mask = target_attn_mask_3)[:, 8:12, :]  #[5, 4, 512 ]
            target_features_fine_final = torch.cat([target_features_fine_0, target_features_fine_1, target_features_fine_2], dim=1)
            cum_dists_fine = self.video_video_classify_head(support_features_fine_final, target_features_fine_final, support_labels) # [query,5]

            # only Fine branch
            cum_dists_visual = cum_dists_fine
            cum_dists_visual_softmax = F.softmax(-cum_dists_visual, dim=1) # 越大越好
            cum_dists_visual_entropy = -torch.sum(cum_dists_visual_softmax * cum_dists_visual_softmax.log(), dim=-1)  # [15]
            cum_dists_visual_entropy = 1 - cum_dists_visual_entropy/torch.log(torch.tensor(5.0))

        elif hasattr(self.args.TRAIN, "TEMPORAL_SEGMENT") and self.args.TRAIN.TEMPORAL_SEGMENT == "CLUSTER":
            # Fine_Attention (2)- CLUSTER_ATTEN

            support_features_fine, support_weights = self.culster(support_video_features, target_len=3) # [5, 12, 512]
            target_features_fine, target_weights = self.culster(target_video_features, target_len=3) # [15, 12, 512]

            support_weights = nn.Parameter(support_weights.cuda(), requires_grad=False)
            target_weights = nn.Parameter(target_weights.cuda(), requires_grad=False)

            len_num = self.args.DATA.NUM_INPUT_FRAMES + 4

            support_attn_masks = []
            for weight in support_weights:
                attn_mask = torch.ger(weight, weight)
                support_attn_masks.append(attn_mask)
            support_attn = nn.Parameter(torch.stack(support_attn_masks, dim=0).cuda(), requires_grad=False) # [5, 12, 12]
            support_attn = support_attn.unsqueeze(1).expand(-1, self.heads, -1, -1).reshape(-1, len_num, len_num) # [N*K*8, 3, 12, 12]
            target_attn_masks = []
            for weight in target_weights:
                attn_mask = torch.ger(weight, weight)
                target_attn_masks.append(attn_mask)
            target_attn = nn.Parameter(torch.stack(target_attn_masks, dim=0).cuda(), requires_grad=False)# [15, 12, 12]
            target_attn = target_attn.unsqueeze(1).expand(-1, self.heads, -1, -1).reshape(-1, len_num, len_num) # [N*query_cls*8, 3, 12, 12]


            if hasattr(self.args.TRAIN, "Long_Para"):
                long_para = self.args.TRAIN.Long_Para
            else:
                long_para = 0.0
            support_attn_mask_1 =  torch.where(support_attn == 1, torch.tensor(1.0, device=support_attn.device), torch.tensor(long_para, device=support_attn.device))
            support_attn_mask_2 =  torch.where(support_attn == 4, torch.tensor(1.0, device=support_attn.device), torch.tensor(long_para, device=support_attn.device))
            support_attn_mask_3 =  torch.where(support_attn == 9, torch.tensor(1.0, device=support_attn.device), torch.tensor(long_para, device=support_attn.device))

            target_attn_mask_1 =  torch.where(target_attn == 1, torch.tensor(1.0, device=target_attn.device), torch.tensor(long_para, device=target_attn.device))
            target_attn_mask_2 =  torch.where(target_attn == 4, torch.tensor(1.0, device=target_attn.device), torch.tensor(long_para, device=target_attn.device))
            target_attn_mask_3 =  torch.where(target_attn == 9, torch.tensor(1.0, device=target_attn.device), torch.tensor(long_para, device=target_attn.device))

            # for Alignment
            support_segment_0 = (support_weights==1).int().unsqueeze(-1).expand(-1,-1,512)*support_features_fine
            support_segment_1 = (support_weights==2).int().unsqueeze(-1).expand(-1,-1,512)*support_features_fine
            support_segment_2 = (support_weights==3).int().unsqueeze(-1).expand(-1,-1,512)*support_features_fine

            support_features_fine_textual_0  = (support_weights==1).int().unsqueeze(-1).expand(-1,-1,512)*(support_features_fine + context_support_fine[:,0,:].unsqueeze(1))
            support_features_fine_textual_1  = (support_weights==2).int().unsqueeze(-1).expand(-1,-1,512)*(support_features_fine + context_support_fine[:,1,:].unsqueeze(1))
            support_features_fine_textual_2  = (support_weights==3).int().unsqueeze(-1).expand(-1,-1,512)*(support_features_fine + context_support_fine[:,2,:].unsqueeze(1))

            support_features_fine_textual = support_features_fine_textual_0 + support_features_fine_textual_1 + support_features_fine_textual_2  # [5, 12, 512]


            support_features_fine_0 = self.fine_attn(support_features_fine_textual, support_features_fine, support_features_fine, attn_mask=support_attn_mask_1) #[5, 12, 512 ]
            support_features_fine_0 = (support_weights==1).int().unsqueeze(-1).expand(-1,-1,512)*support_features_fine_0
            support_features_fine_1 = self.fine_attn(support_features_fine_textual, support_features_fine, support_features_fine, attn_mask=support_attn_mask_2)
            support_features_fine_1 = (support_weights==2).int().unsqueeze(-1).expand(-1,-1,512)*support_features_fine_1
            support_features_fine_2 = self.fine_attn(support_features_fine_textual, support_features_fine, support_features_fine, attn_mask=support_attn_mask_3)
            support_features_fine_2 = (support_weights==3).int().unsqueeze(-1).expand(-1,-1,512)*support_features_fine_2
            support_features_fine_final = support_features_fine_0 + support_features_fine_1 + support_features_fine_2 #[5, 12, 512]

            target_features_fine_0 = self.fine_attn(target_features_fine, target_features_fine, target_features_fine, attn_mask=target_attn_mask_1)
            target_features_fine_0 = (target_weights==1).unsqueeze(-1).expand(-1,-1,512)*target_features_fine_0
            target_features_fine_1 = self.fine_attn(target_features_fine, target_features_fine, target_features_fine, attn_mask=target_attn_mask_2)
            target_features_fine_1 = (target_weights==2).unsqueeze(-1).expand(-1,-1,512)*target_features_fine_1
            target_features_fine_2 = self.fine_attn(target_features_fine, target_features_fine, target_features_fine, attn_mask=target_attn_mask_3)
            target_features_fine_2 = (target_weights==3).unsqueeze(-1).expand(-1,-1,512)*target_features_fine_2
            target_features_fine_final = target_features_fine_0 + target_features_fine_1 + target_features_fine_2 #[15, 12, 512]


            cum_dists_fine = self.video_video_classify_head(support_features_fine_final, target_features_fine_final, support_labels, support_weights, target_weights) # [query,5]
            # cum_dists_fine_update = self.video_video_classify_head_update(support_features_fine, target_features_fine, support_labels, support_weights, target_weights) # [query,5]

            # only Fine branch
            cum_dists_visual = cum_dists_fine
            cum_dists_visual_softmax = F.softmax(-cum_dists_visual, dim=1) #
            cum_dists_visual_entropy = -torch.sum(cum_dists_visual_softmax * cum_dists_visual_softmax.log(), dim=-1)  # [15]
            cum_dists_visual_entropy = 1 - cum_dists_visual_entropy/torch.log(torch.tensor(5.0))


        if not self.training and hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT:
            if hasattr(self.args.TRAIN, "TEXT_MATCH") and self.args.TRAIN.TEXT_MATCH == "GLOBAL":
                # Text Mean
                text_features = text_features_global.mean(1)
                image_features = target_video_features.mean(1)
                image_features = self.classification_layer(image_features)
                image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
                # cosine similarity as logits
                cum_dists_text = image_features_norm @ text_features_norm.t()  # [15, 5] 相似度越高 越接近
                # video-text matching softmax
                cum_dists_text_softmax = F.softmax(cum_dists_text, dim=1)

            if hasattr(self.args.TRAIN, "TEXT_MATCH") and self.args.TRAIN.TEXT_MATCH == "MEAN":
                # Text Mean
                text_features = text_features_fine.mean(1)
                image_features = target_video_features.mean(1)
                image_features = self.classification_layer(image_features)
                image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
                # cosine similarity as logits
                cum_dists_text = image_features_norm @ text_features_norm.t()  # [15, 5] 相似度越高 越接近
                # video-text matching softmax
                cum_dists_text_softmax = F.softmax(cum_dists_text, dim=1)

            elif hasattr(self.args.TRAIN, "TEXT_MATCH") and self.args.TRAIN.TEXT_MATCH == "FINEACTION":
                # Text- FineAction
                target_segment_0 = ((target_weights==1).int().unsqueeze(-1).expand(-1,-1,512)*target_features_fine).sum(dim=1)/(target_weights==1).sum(dim=1).unsqueeze(-1)
                text_features_0 = text_features_fine[:,0,:]
                target_segment_0_norm = target_segment_0 / target_segment_0.norm(dim=1, keepdim=True)
                text_features_0_norm = text_features_0 / text_features_0.norm(dim=1, keepdim=True)
                cum_dists_text_0 = target_segment_0_norm @ text_features_0_norm.t()  # [15, 5] 相似度越高 越接近

                target_segment_1 = ((target_weights==2).int().unsqueeze(-1).expand(-1,-1,512)*target_features_fine).sum(dim=1)/(target_weights==2).sum(dim=1).unsqueeze(-1)
                text_features_1 = text_features_fine[:,1,:]
                target_segment_1_norm = target_segment_1 / target_segment_1.norm(dim=1, keepdim=True)
                text_features_1_norm = text_features_1 / text_features_1.norm(dim=1, keepdim=True)
                cum_dists_text_1 = target_segment_1_norm @ text_features_1_norm.t()  # [15, 5] 相似度越高 越接近

                target_segment_2 = ((target_weights==3).int().unsqueeze(-1).expand(-1,-1,512)*target_features_fine).sum(dim=1)/(target_weights==3).sum(dim=1).unsqueeze(-1)
                text_features_2 = text_features_fine[:,2,:]
                target_segment_2_norm = target_segment_2 / target_segment_2.norm(dim=1, keepdim=True)
                text_features_2_norm = text_features_2 / text_features_2.norm(dim=1, keepdim=True)
                cum_dists_text_2 = target_segment_2_norm @ text_features_2_norm.t()  # [15, 5] 相似度越高 越接近

                # 直接结果相加作为最终分类结果
                dum_dist_text = cum_dists_text_0 + cum_dists_text_1 + cum_dists_text_2
                cum_dists_text_softmax = F.softmax(dum_dist_text, dim=1)

                # # 选择分类结果最大的Fine Action 作为最终分类结果
                # cum_dists_text_softmax_0 = F.softmax(cum_dists_text_0, dim=1)
                # cum_dists_text_softmax_1 = F.softmax(cum_dists_text_1, dim=1)
                # cum_dists_text_softmax_2 = F.softmax(cum_dists_text_2, dim=1)

                # # 堆叠分类结果，形状 [3, batch_size, num_classes]
                # cum_dists_text_softmax = torch.stack([cum_dists_text_softmax_0, cum_dists_text_softmax_1, cum_dists_text_softmax_2], dim=0)

                # max_probs_per_result = cum_dists_text_softmax.max(dim=2).values
                # # print(max_probs_per_result
                # best_result_indices = max_probs_per_result.argmax(dim=0)
                # # 从堆叠的分类结果中选择最终结果
                # cum_dists_text_softmax = cum_dists_text_softmax[best_result_indices, torch.arange(target_bs)]

            # cum_dists = - ((1-visual_weight)*cum_dists_text_softmax + visual_weight*cum_dists_visual_softmax) #越小越好
            if hasattr(self.args.TEST, "VISUAL_WEIGHT") and self.args.TEST.VISUAL_WEIGHT:
                visual_weight = self.args.TEST.VISUAL_WEIGHT
            else:
                visual_weight = 0.025
            cum_dists = cum_dists_text_softmax.pow(1-visual_weight)*cum_dists_visual_softmax.pow(visual_weight)
            cum_dists = - cum_dists

        else:
            cum_dists =  cum_dists_visual

        #================================Contrastive Losss========================================
        # support_features_prototype = [torch.mean(torch.index_select(support_features_fine_final, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        # support_features_fine_final = torch.stack(support_features_prototype)

        support_features_global = support_features_fine_final.mean(dim=1) # [5,512]
        target_features_global = target_features_fine_final.mean(dim=1) # [query_num,512]

        # support to query
        class_sim_s2q = cos_sim(support_features_fine_final,target_features_global)  # [5, frame_num, query_num]
        class_dists_s2q = 1 - class_sim_s2q #[0,2] #[5,12,query_num]
        class_dists_s2q = [torch.sum(torch.index_select(class_dists_s2q, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q = torch.stack(class_dists_s2q).squeeze() # [5, query_num]
        class_dists_s2q = rearrange(class_dists_s2q * self.scale, 'c q -> q c')

        # query to support
        class_sim_q2s = cos_sim(target_features_fine_final, support_features_global)  # [query_num, frame_num, 5]
        class_dists_q2s = 1 - class_sim_q2s   #[0,2]  #[query_num,12,5]
        class_dists_q2s = [torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze() # [5, query_num]
        class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')
        #================================Contrastive Losss========================================

        class_dists = cum_dists

        # class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        # class_dists = torch.stack(class_dists)
        # class_dists = rearrange(class_dists, 'c q -> q c')

        return_dict = {'logits': - class_dists, "class_logits": class_text_logits, "logits_s2q": -class_dists_s2q, "logits_q2s": -class_dists_q2s}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


def classname2words(name):
    name_dict = {
        'Shotput': 'shot put', 'PoleVault': 'pole vault', 'BabyCrawling': 'baby crawling', 'BoxingPunchingBag': 'boxing punching bag', 'Typing': 'typing', 'BalanceBeam': 'balance beam', 'HandstandPushups': 'handstand pushups',
        'Diving': 'diving', 'BlowingCandles': 'blowing candles', 'PlayingViolin': 'playing violin', 'UnevenBars': 'uneven bars', 'Hammering': 'hammering', 'Skiing': 'skiing', 'WallPushups': 'wall pushups',
        'SkateBoarding': 'skateBoarding', 'Nunchucks': 'nunchucks', 'Punch': 'punch', 'FrisbeeCatch': 'frisbee catch', 'CuttingInKitchen': 'cut in kitchen', 'Rowing': 'rowing', 'HorseRiding': 'horse riding',
        'SumoWrestling': 'sumo wrestling', 'CricketShot': 'cricket shot', 'FieldHockeyPenalty': 'field hockey penalty', 'ParallelBars': 'parallel bars', 'Biking': 'biking', 'TennisSwing': 'tennis swing', 'SkyDiving': 'sky diving',
        'PizzaTossing': 'pizza tossing', 'BandMarching': 'band marching', 'SalsaSpin': 'salsa spin', 'WritingOnBoard': 'writing on board', 'BlowDryHair': 'blow dry hair', 'PlayingSitar': 'playing sitar',
        'Surfing': 'surfing', 'Billiards': 'billiards', 'MilitaryParade': 'military parade', 'PlayingFlute': 'playing flute', 'PlayingDhol': 'playing dhol', 'BaseballPitch': 'baseball pitch', 'GolfSwing': 'golf swing', 'Rafting': 'rafting',
        'PlayingPiano': 'playing piano', 'Drumming': 'drumming', 'StillRings': 'still rings', 'LongJump': 'long jump', 'Basketball': 'basketball', 'PlayingGuitar': 'playing guitar', 'FloorGymnastics': 'floor gymnastics',
        'SoccerJuggling': 'soccer juggling', 'Lunges': 'lunges', 'JumpingJack': 'jumping jack', 'JavelinThrow': 'javelin throw', 'IceDancing': 'ice dancing', 'HorseRace': 'horse race', 'Mixing': 'mixing', 'BasketballDunk': 'basketball dunk',
        'BoxingSpeedBag': 'boxing speed bag', 'PushUps': 'pushups', 'PommelHorse': 'pommel horse', 'HulaHoop': 'Hula hoop', 'RockClimbingIndoor': 'rock climbing indoor', 'PlayingDaf': 'playing daf drum', 'HandstandWalking': 'handstand walking',
        'JumpRope': 'jump rope', 'ShavingBeard': 'shaving beard', 'Kayaking': 'kayaking', 'HighJump': 'high jump', 'Swing': 'play on a swing', 'Fencing': 'fencing', 'CricketBowling': 'cricket bowling', 'BodyWeightSquats': 'body weight squats',
        'HeadMassage': 'head massage', 'HammerThrow': 'hammer throw', 'TableTennisShot': 'table tennis shot', 'Skijet': 'Jet Ski', 'RopeClimbing': 'rope climbing', 'WalkingWithDog': 'walk dog', 'BenchPress': 'bench press', 'TaiChi': 'tai chi',
        'VolleyballSpiking': 'volleyball spiking', 'YoYo': 'yoyo', 'PlayingTabla': 'playing tabla', 'FrontCrawl': 'front crawl', 'MoppingFloor': 'mopping floor', 'Knitting': 'knitting', 'ThrowDiscus': 'throw disc', 'ApplyEyeMakeup': 'apply eye makeup',
        'ApplyLipstick': 'apply lipstick', 'PlayingCello': 'playing cello', 'BrushingTeeth': 'brushing teeth', 'CleanAndJerk': 'weight lift', 'Haircut': 'hair cut', 'PullUps': 'pullups', 'BreastStroke': 'breaststroke',
        'SoccerPenalty': 'soccer penalty', 'TrampolineJumping': 'trampoline jumping', 'CliffDiving': 'cliff diving', 'JugglingBalls': 'juggling balls', 'Archery': 'archery', 'Bowling': 'bowling',

        'clean and jerk': 'weight lift', 'dancing gangnam style': 'dance korean', 'breading or breadcrumbing': 'bread crumb', 'faceplanting': 'face fall', 'hoverboarding': 'skateboard electric', 'hurling (sport)': 'hurl sport',

        "small_train0": "Pouring [something] into [something]",
        "small_train1": "Poking a stack of [something] without the stack collapsing",
        "small_train2": "Pretending to poke [something]",
        "small_train3": "Lifting up one end of [something] without letting it drop down",
        "small_train4": "Moving [part] of [something]",
        "small_train5": "Moving [something] and [something] away from each other",
        "small_train6": "Removing [something], revealing [something] behind",
        "small_train7": "Plugging [something] into [something]",
        "small_train8": "Tipping [something] with [something in it] over, so [something in it] falls out",
        "small_train9": "Stacking [number of] [something]",
        "small_train10": "Putting [something] onto a slanted surface but it doesn't glide down",
        "small_train11": "Moving [something] across a surface until it falls down",
        "small_train12": "Throwing [something] in the air and catching it",
        "small_train13": "Putting [something that cannot actually stand upright] upright on the table, so it falls on its side",
        "small_train14": "Holding [something] next to [something]",
        "small_train15": "Pretending to put [something] underneath [something]",
        "small_train16": "Poking [something] so lightly that it doesn't or almost doesn't move",
        "small_train17": "Approaching [something] with your camera",
        "small_train18": "Poking [something] so that it spins around",
        "small_train19": "Pushing [something] so that it falls off the table",
        "small_train20": "Spilling [something] next to [something]",
        "small_train21": "Pretending or trying and failing to twist [something]",
        "small_train22": "Pulling two ends of [something] so that it separates into two pieces",
        "small_train23": "Lifting up one end of [something], then letting it drop down",
        "small_train24": "Tilting [something] with [something] on it slightly so it doesn't fall down",
        "small_train25": "Spreading [something] onto [something]",
        "small_train26": "Touching (without moving) [part] of [something]",
        "small_train27": "Turning the camera left while filming [something]",
        "small_train28": "Pushing [something] so that it slightly moves",
        "small_train29": "Uncovering [something]",
        "small_train30": "Moving [something] across a surface without it falling down",
        "small_train31": "Putting [something] behind [something]",
        "small_train32": "Attaching [something] to [something]",
        "small_train33": "Pulling [something] onto [something]",
        "small_train34": "Burying [something] in [something]",
        "small_train35": "Putting [number of] [something] onto [something]",
        "small_train36": "Letting [something] roll along a flat surface",
        "small_train37": "Bending [something] until it breaks",
        "small_train38": "Showing [something] behind [something]",
        "small_train39": "Pretending to open [something] without actually opening it",
        "small_train40": "Pretending to put [something] onto [something]",
        "small_train41": "Moving away from [something] with your camera",
        "small_train42": "Wiping [something] off of [something]",
        "small_train43": "Pretending to spread air onto [something]",
        "small_train44": "Holding [something] over [something]",
        "small_train45": "Pretending or failing to wipe [something] off of [something]",
        "small_train46": "Pretending to put [something] on a surface",
        "small_train47": "Moving [something] and [something] so they collide with each other",
        "small_train48": "Pretending to turn [something] upside down",
        "small_train49": "Showing [something] to the camera",
        "small_train50": "Dropping [something] onto [something]",
        "small_train51": "Pushing [something] so that it almost falls off but doesn't",
        "small_train52": "Piling [something] up",
        "small_train53": "Taking [one of many similar things on the table]",
        "small_train54": "Putting [something] in front of [something]",
        "small_train55": "Laying [something] on the table on its side, not upright",
        "small_train56": "Lifting a surface with [something] on it until it starts sliding down",
        "small_train57": "Poking [something] so it slightly moves",
        "small_train58": "Putting [something] into [something]",
        "small_train59": "Pulling [something] from right to left",
        "small_train60": "Showing that [something] is empty",
        "small_train61": "Spilling [something] behind [something]",
        "small_train62": "Letting [something] roll down a slanted surface",
        "small_train63": "Holding [something] behind [something]",
        "small_test0": "Twisting (wringing) [something] wet until water comes out",
        "small_test1": "Poking a hole into [something soft]",
        "small_test2": "Pretending to take [something] from [somewhere]",
        "small_test3": "Putting [something] upright on the table",
        "small_test4": "Poking a hole into [some substance]",
        "small_test5": "Rolling [something] on a flat surface",
        "small_test6": "Poking a stack of [something] so the stack collapses",
        "small_test7": "Twisting [something]",
        "small_test8": "[Something] falling like a feather or paper",
        "small_test9": "Putting [something] on the edge of [something] so it is not supported and falls down",
        "small_test10": "Pushing [something] off of [something]",
        "small_test11": "Dropping [something] into [something]",
        "small_test12": "Letting [something] roll up a slanted surface, so it rolls back down",
        "small_test13": "Pushing [something] with [something]",
        "small_test14": "Opening [something]",
        "small_test15": "Putting [something] on a surface",
        "small_test16": "Taking [something] out of [something]",
        "small_test17": "Spinning [something] that quickly stops spinning",
        "small_test18": "Unfolding [something]",
        "small_test19": "Moving [something] towards the camera",
        "small_test20": "Putting [something] next to [something]",
        "small_test21": "Scooping [something] up with [something]",
        "small_test22": "Squeezing [something]",
        "small_test23": "Failing to put [something] into [something] because [something] does not fit",

        "full_train0": "Bending [something] until it breaks",
        "full_train1": "Closing [something]",
        "full_train2": "Covering [something] with [something]",
        "full_train3": "Dropping [something] behind [something]",
        "full_train4": "Dropping [something] in front of [something]",
        "full_train5": "Dropping [something] into [something]",
        "full_train6": "Folding [something]",
        "full_train7": "Holding [something]",
        "full_train8": "Holding [something] next to [something]",
        "full_train9": "Letting [something] roll along a flat surface",
        "full_train10": "Letting [something] roll down a slanted surface",
        "full_train11": "Lifting a surface with [something] on it but not enough for it to slide down",
        "full_train12": "Lifting [something] with [something] on it",
        "full_train13": "Moving away from [something] with your camera",
        "full_train14": "Moving [something] across a surface until it falls down",
        "full_train15": "Moving [something] and [something] closer to each other",
        "full_train16": "Moving [something] and [something] so they collide with each other",
        "full_train17": "Moving [something] down",
        "full_train18": "Moving [something] up",
        "full_train19": "Plugging [something] into [something]",
        "full_train20": "Poking a hole into [something soft]",
        "full_train21": "Poking [something] so lightly that it doesn't or almost doesn't move",
        "full_train22": "Poking [something] so that it falls over",
        "full_train23": "Pouring [something] into [something]",
        "full_train24": "Pouring [something] into [something] until it overflows",
        "full_train25": "Pouring [something] onto [something]",
        "full_train26": "Pretending to be tearing [something that is not tearable]",
        "full_train27": "Pretending to close [something] without actually closing it",
        "full_train28": "Pretending to pick [something] up",
        "full_train29": "Pretending to put [something] next to [something]",
        "full_train30": "Pretending to spread air onto [something]",
        "full_train31": "Pretending to take [something] out of [something]",
        "full_train32": "Pulling [something] onto [something]",
        "full_train33": "Pulling two ends of [something] so that it gets stretched",
        "full_train34": "Pulling two ends of [something] so that it separates into two pieces",
        "full_train35": "Pushing [something] from left to right",
        "full_train36": "Pushing [something] off of [something]",
        "full_train37": "Pushing [something] so that it falls off the table",
        "full_train38": "Pushing [something] so that it slightly moves",
        "full_train39": "Putting [number of] [something] onto [something]",
        "full_train40": "Putting [something] and [something] on the table",
        "full_train41": "Putting [something] onto a slanted surface but it doesn't glide down",
        "full_train42": "Putting [something] onto [something]",
        "full_train43": "Putting [something similar to other things that are already on the table]",
        "full_train44": "Showing a photo of [something] to the camera",
        "full_train45": "Showing [something] behind [something]",
        "full_train46": "[Something] colliding with [something] and both are being deflected",
        "full_train47": "Spilling [something] next to [something]",
        "full_train48": "Spilling [something] onto [something]",
        "full_train49": "Spinning [something] that quickly stops spinning",
        "full_train50": "Spreading [something] onto [something]",
        "full_train51": "Squeezing [something]",
        "full_train52": "Stuffing [something] into [something]",
        "full_train53": "Taking [something] from [somewhere]",
        "full_train54": "Tearing [something] into two pieces",
        "full_train55": "Tilting [something] with [something] on it slightly so it doesn't fall down",
        "full_train56": "Tilting [something] with [something] on it until it falls off",
        "full_train57": "Tipping [something] with [something in it] over, so [something in it] falls out",
        "full_train58": "Turning the camera downwards while filming [something]",
        "full_train59": "Turning the camera left while filming [something]",
        "full_train60": "Turning the camera upwards while filming [something]",
        "full_train61": "Twisting (wringing) [something] wet until water comes out",
        "full_train62": "Twisting [something]",
        "full_train63": "Uncovering [something]",
        "full_test0": "Approaching [something] with your camera",
        "full_test1": "Digging [something] out of [something]",
        "full_test2": "Dropping [something] next to [something]",
        "full_test3": "Dropping [something] onto [something]",
        "full_test4": "Failing to put [something] into [something] because [something] does not fit",
        "full_test5": "Lifting up one end of [something] without letting it drop down",
        "full_test6": "Picking [something] up",
        "full_test7": "Poking a stack of [something] without the stack collapsing",
        "full_test8": "Pouring [something] out of [something]",
        "full_test9": "Pretending to open [something] without actually opening it",
        "full_test10": "Pretending to put [something] behind [something]",
        "full_test11": "Pretending to put [something] into [something]",
        "full_test12": "Pretending to put [something] underneath [something]",
        "full_test13": "Pretending to sprinkle air onto [something]",
        "full_test14": "Pulling [something] from left to right",
        "full_test15": "Pulling [something] out of [something]",
        "full_test16": "Pushing [something] from right to left",
        "full_test17": "Removing [something], revealing [something] behind",
        "full_test18": "Showing [something] next to [something]",
        "full_test19": "Showing that [something] is empty",
        "full_test20": "Spilling [something] behind [something]",
        "full_test21": "Taking [something] out of [something]",
        "full_test22": "Throwing [something] in the air and letting it fall",
        "full_test23": "Tipping [something] over"
    }
    if isinstance(name, str):
        if name in name_dict:
            name = name_dict[name].replace(']', '').replace('[', '').replace('_', ' ')
        return name
    if isinstance(name, list):
        words_list = []
        for label in name:
            if isinstance(label, list):
                label = label[0]    # because [['WritingOnBoard'], ['SumoWrestling'], ['SumoWrestling']...]
            if label in name_dict:
                label_words = name_dict[label].replace(']', '').replace('[', '').replace('_', ' ')
            else:
                label_words = label.replace(']', '').replace('[', '').replace('_', ' ')
            words_list.append(label_words)
        return words_list


def hard_prompt(cls_words, temp=None):
    if temp is None:
        temp = 'Play a human action of {}.'  # 'An action video of {}.'
    if isinstance(cls_words, str):
        return temp.format(cls_words)
    if isinstance(cls_words, list):
        text_list = []
        for cls in cls_words:
            text_list.append(temp.format(cls))
        return text_list