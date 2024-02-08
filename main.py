import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import ssl  # 导入证书模块



# 设置为False跳过Note执行(例如调试)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class EncoderDecoder(nn.Module):
    """
    一个标准的解码器，编码器模型
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "接收处理屏蔽的src和目标序列"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """
    定义一个标准的线性+softmax生成步骤。
    说人话，这个是用来接受最后的decode的结果，并且返回词典中每个词的概率
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        "为什么用log_softmax，主要是梯度和计算速度的考虑，可以百度一下，资料很多"
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """克隆N层一模一样的
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """每一个小的编码器的核心结构,由传入层(及其个数，后文使用的层个数是2)和层归一化组成
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "做前向传播需要依次传入每一个层，并且带上掩码"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "构造一个'层归一化'模块"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # -1 表示计算沿着张量的最后一维度的平均值。keepdim 是一个布尔值，用于指定是否保留维度。
        # 如果将 keepdim 设置为 True，则输出张量的形状将与输入张量的形状相同，只是最后一维的大小为 1。
        # 如果将 keepdim 设置为 False，则输出张量的形状将不包括最后一维。
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    紧跟在层归一化模块后的残差链接
    注意，为了简化代码，先用norm而不是最后才使用(注意图1是sublayer计算后才norm)。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "将残差层应用在所有大小相同的层"
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder 由自注意力层和前向网络层构成"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "同图1左边的链接所示"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "一个带掩码的N层解码器通用结构"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "解码器是的自注意力模块是由编码器和解码器的att共同构成，再加上前馈网络"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 和编码器一样，共享SublayerConnection的结构，这个结构包括正则化和dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "图1右边所示的解码器结构即下面的代码"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 解码器第二个attn的k,v是编码器提供的输出，用编码器的x去查解码器的attn输出。
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "屏蔽后面的位置"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def example_mask():
    # 第一眼看这个嵌套循环给看懵了。其实就是用两个for循环生成了一个二维坐标，每一个都是一个df对象
    # 看下面这个就好理解了
    # 其实:=[(x,y) for y in range(20) for x in range(20)]

    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )

def attention(query, key, value, mask=None, dropout=None):
    "计算 '缩放点积注意力'"
    # 返回Query最后一个轴的长度，即d_k
    d_k = query.size(-1)
    # key.transpose实际上做的就是转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果存在掩码，使用掩码计算得分
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 按照最后一个轴计算softmax,即按照每行内进行softmax
    p_attn = scores.softmax(dim=-1)
    # 如果dropout存在则进行dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后和V相乘返回V的得分
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "接收多头的个数和维度进行初始化"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设d_v总是等于d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "实现图2"
        if mask is not None:
            # 对每一个头都用相同的掩码
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 这一段原文的投影应该指的就是图里面第一个块后面那些阴影，其实就是多头
        # 1) 批量计算线性投影从 d_model => h x d_k
        # 这一段很鸡贼，给了四个前馈线性层网络，这段打包只用了前三个，最后一个前馈网络什么时候用呢？最后return用。
        # 分别把query,key,value传给这三个线性层，然后reshape(这里用的view)成多头。然后得到了query,key,value
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算attention
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) 在最后的线性层使用一个视图进行"Concat"，相当于把之前的多头变成单头
        # 关于调用contiguous原因 https://blog.csdn.net/weixin_43332715/article/details/124749348：
        # 1 transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy；
        # 2 维度变换后的变量是之前变量的浅拷贝，指向同一区域，即view操作会连带原来的变量一同变形，这是不合法的，所以也会报错；---- 这个解释有部分道理，也即contiguous返回了tensor的深拷贝contiguous copy数据；

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value

        # 这里，用到了最后一个linear
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "实现一个FFN模型"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "实现PE(位置编码)函数"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算位置编码。
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # requires_grad_(False)：禁用梯度下降
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "帮助: 从超参数中构建一个模型"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 这里很重要
    # 用Glorot/fan_avg初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    # 这里用LongTensor是因为怕单词长度过长。
    # torch.LongTensor 是一种整数类型的张量，它可以存储长整型数据。这种类型的张量通常用于存储索引数据，例如词汇表中的单词索引。
    # torch.Tensor 是一种浮点类型的张量，它可以存储单精度或双精度浮点数据。这种类型的张量通常用于存储神经网络中的权重参数或输入/输出数据。
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # shape=(1,1,10)，元素全部为1的mask
    src_mask = torch.ones(1, 1, 10)
    # 把encode输出的中间输出保存到memory，encode一步完成，不需要循环
    memory = test_model.encode(src, src_mask)
    # 创建一个和输入的张量类型一致的目标结果ys张量，一开始只有0起始符，所以shape=(1,1)，随着预测，shape=(1,n)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        # 传入memory,源mask,ys
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 生成概率矩阵
        prob = test_model.generator(out[:, -1])
        # 取值和索引，我们只需要索引
        _, next_word = torch.max(prob, dim=1)
        # 取出单个词
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()

class Batch:
    """训练期间用于保存一批带掩码的数据的对象"""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        # 关于下面这一段：
        # (torch.tensor([[ 0, 2, 4, 5, 1, 0, 2 ]]) != 2).unsqueeze(-2)
        # print：tensor([[[ True, False,  True,  True,  True,  True, False]]])
        # 实际是把2元素打上掩码。
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 下面两个分别去掉句子的开始符和结束符，这两个符合不参与运算
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            # 把输入指定位置的下三角掩码
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个掩码来隐藏并填充未来的word"
        # 和src一样，需要把<blank>符合盖住
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 取&操作
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
class TrainState:
    """用来跟踪当前训练的情况，包括步数，梯度步数，样本使用数和已经处理的tokens数量"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """一个训练epoch
    data_iter: 可迭代对象，一次返回一个Batch对象或者加上索引
    model:训练的模型，这里就是Transformer
    loss_compute: SimpleLossCompute对象，用于计算损失
    optimizer: 优化器。这里是Adam优化器。验证时，optimizer是DummyOptimizer。DummyOptimizer不会真的更新模型参数，主要用于不同优化器效果的对比。
    scheduler：执行控制器。scheduler是一种用于调整优化器学习率的工具。 它可以帮助我们在训练过程中根据指定的策略调整学习率，
      以提高模型的性能这里是LambdaLR对象，用于调整Adam的学习率，实现WarmUp；验证时，scheduler是DummyScheduler。
    accum_iter: 每迭代n个batch更新一次模型的参数。这里默认n=1，就是每次batch都更新参数。
    train_state: TrainState对象，用于保存前训练的情况
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        # 注意这里的out是decoder输出的结果，这会还没有经过最后一层linear+softmax
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # 这里才传入out和训练目标tgt_y计算了loss和loss_node。loss_node返回的是正则化的损失；
        # loss用来计算损失，loss_node用来梯度下降更新参数
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        # 只有在train或者train+log的模式才开启参数更新
        if mode == "train" or mode == "train+log":
            # 先通过backward计算出来梯度
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                # 调用依次梯度下降
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            # 我们在备注里提到过,scheduler的作用就是用来优化学习，控制学习率等超参数。这里调用step就是更新学习率相关的参数
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # 下面是每40个epoch打印下相关日志
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # 学习率
            lr = optimizer.param_groups[0]["lr"]
            # 40个epoch花费的时间
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    """
    我们必须将LambdaLR函数的最小步数默认为1。以避免零点导致的负数学习率。
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # 在我们的配置列表里有三个不同的案例.
    for idx, example in enumerate(opts):
        # 创建一个Adam优化器
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        # 这里创建一个神经网络学习调度器，优化器就是刚才创建的Adam优化器，lr(学习率)的调整函数就是上面的rate函数，根据当前的step来调整。
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # 采取20K假训练步骤，保存每一步的学习率
        for step in range(20000):
            # 把当前的学习率追加到等会输出的list
            tmp.append(optimizer.param_groups[0]["lr"])
            # optimizer.step()是更新网络的参数
            optimizer.step()
            # scheduler.step()是更新学习率等控制网络学习的参数
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # 关掉最大限制，使得altair能够处理超过5000行的数据
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


class LabelSmoothing(nn.Module):
    "实现标签平滑."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 定义一个KL散度loss网络，损失的计算方式是sum，求和
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # 克隆一份作为真实分布
        true_dist = x.data.clone()
        # 用smoothing/(self.size - 2) 填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 用confidence填充指定位置的数据，scatter_用法参考[8]
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # detach，把变量从计算图分离，可参考 https://zhuanlan.zhihu.com/p/389738863
        return self.criterion(x, true_dist.clone().detach())


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    # 注意这里传入0，log会导致矩阵出现负无穷，可以调整成1e-9
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    # 如有疑惑，看上面段落解释
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )

def data_gen(V, batch_size, nbatches):
    "为src-tgt复制任务生成一组随机的数据"
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        # 迭代器语法，yield
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "一个简单的损失计算和训练函数"

    def __init__(self, generator, criterion):
        """
        generator: Generator对象，用于根据Decoder的输出预测token
        criterion: LabelSmoothing对象，用于对Label进行平滑处理和损失计算
        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        # 这里顺便用最简单的例子展示了smoothing的用法
        # 先把decoder的x输入generator，得到预测的x
        # 然后把x和预测的y传入，criterion会对y做平滑处理，需要注意的是:
        # 这里传入的y展开成了一个一阶张量，即向量，因为在criterion内部会对它打包，会为每个单词生成一个概率向量
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )

        # 这里又在搞事情，相当于第一个没有norm,第二个sloss是norm版本的，除以的是一个常量,batch.ntokens
        return sloss.data * norm, sloss
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        # Transformer作者开启了下面的验证代码，译者实在没看懂他想验证什么，所以默认取消了，如果想要保持和原作者一样可以取消下面代码注释
        # #####################取消下面注释##################### #
        # model.eval()
        # run_epoch(
        #     data_gen(V, batch_size, 5),
        #     model,
        #     SimpleLossCompute(model.generator, criterion),
        #     DummyOptimizer(),
        #     DummyScheduler(),
        #     mode="eval",
        # )[0]
        # #####################取消上面注释##################### #

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    """对text进行分词
    :param text: 要分词的文本，例如'I love you'
    :param tokenizer: 分词模型，例如：spacy_en
    :return: 分词结果，例如 ["I", "love", "you"]
    """
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    """yield一个Token List
    :param data_iter: 包含句子对的可迭代对象。例如：[("I love you", "我爱你"), ...]
    :param tokenizer: 分词模型。例如spacy_en
    :param index: 要对句子对儿的哪个语言进行分词，例如0表示对上例的英文进行分词
    :return: yield本轮的分词结果，例如['I', 'love', 'you']
    """
    print(type(data_iter))
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    """
    构建德语词典和英语词典
    :return: 返回德语词典和英语词典，均为：Vocab对象
             Vocab对象官方地址为：https://pytorch.org/text/stable/vocab.html#vocab
    """

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    # [9]build_vocab_from_iterator()函数
    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(root="./data",language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(root="./data",language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    ssl._create_default_https_context = ssl._create_unverified_context
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    """返回真正的训练批次张量，并且文本被['i', 'love', 'you']处理成[3,4,5]数字张量。
    :param batch: 一个batch的语句对。例如：
                  [('Ein Kleinkind ...', 'A toddler in ...'), # [(德语), (英语)
                   ....                                       # ...
                   ...]                                       # ... ]
    :param src_pipeline: 德语分词器，也就是tokenize_de方法，后面会定义其实就是对spacy_de的封装
    :param tgt_pipeline: 英语分词器，也就是tokenize_en方法
    :param src_vocab: 德语词典，Vocab对象
    :param tgt_vocab: 英语词典，Vocab对象
    :param device: cpu或cuda
    :param max_padding: 句子的长度。pad长度不足的句子和裁剪长度过长的句子，目的是让不同长度的句子可以组成一个tensor
    :param pad_id: '<blank>'在词典中对应的index
    :return: src和tgt。处理后并batch后的句子。例如：
             src为：[[0, 4354, 314, ..., 1, 2, 2, ..., 2],  [0, 4905, 8567, ..., 1, 2, 2, ..., 2]]
             其中0是<bos>, 1是<eos>, 2是<blank>；src的Shape为(batch_size, max_padding)；tgt同理。
    """
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # 警告 - 覆盖padding - len的负值的值
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
):
    """创建一个dataloaders，实际上返回了两个，一个训练集，一个验证集的
    :param device: cpu或cuda
    :param vocab_src: 源词典，本例中为德语词典
    :param vocab_tgt: 目标词典，本例中为英语词典
    :param spacy_de: 德语分词器
    :param spacy_en: 英语分词器
    :param batch_size: 每个批次的样本量
    :param max_padding: 句子的最大长度。也是需要填充的长度。

    :return: 训练集dataloaders，验证集dataloaders
    """

    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    # 这里对sampler开启了分布式采样
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )
    # [11]torch.utils.data.Dataloader()
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        # 具体参考Q&A[10]
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        # DDP->torch.nn.parallel.DistributedDataParallel:
        # 用于在分布式训练环境中在多个GPU或机器上并行训练一个模型。
        # 它的工作原理是在每个GPU或机器上复制模型，并在训练期间使用通信后端来同步模型的梯度和缓冲区。
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # 创建一个标签平滑处理模型
    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    # 使用指定的GPU
    criterion.cuda(gpu)
    # 创建Dataloaders
    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    """分布式GPU训练的主入口
    """
    # 译者把源代码下面这一段屏蔽了，train_worker的定义就在上面一个Block，不需要导入了。
    # from the_annotated_transformer import train_worker

    # mp: 就是import torch.multiprocessing as mp中的多进程启动
    # train_worker：每个训练的入口
    # nprocs：需要启动的进程数
    # args: 每个进程都会收到的参数
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    # spawn从头构建一个子进程，父进程的数据等拷贝到子进程空间内，拥有自己的Python解释器，
    # 所以需要重新加载一遍父进程的包，因此启动较慢，由于数据都是自己的，安全性较高
    # 可参考：https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
    # spawn会将train_worker以(i, args)形式调用，i是master为每个子进程自动分配，所以不用传。
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    # distributed是用来控制是否分布式训练，如果开启分布式训练，就走第一个if
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    # 不启用分布式就直接调用train_worker
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,  # 这个参数控制是否启动分布式训练
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    # model保存的位置
    model_path = "multi30k_model_final.pt"

    # 如果模型不存在，则调用train_model先训练。这里传入的前四个参数都是"准备数据"章节的全局变量。
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    # 使用词典创建模型
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    # 加载模型参数
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


if is_interactive_notebook():
    model = load_trained_model()