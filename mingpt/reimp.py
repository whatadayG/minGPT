import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# TODO: factor these out into problem-specific and default (GPT paper implied) configs using a slightly better class
ToyProblemConfig = Config(
        # took the following three from Karpathy's demo
        num_decoder_blocks = 3,
        num_attention_heads = 3,
        num_hidden_neurons = 48,

        max_sequence_length = 11,
        num_distinct_tokens = 3,
        word_embedding_method = "learned",
        pos_embedding_method = "learned",
        embedding_dimension = 48, 
        linear_layer_dimension = 48 * 4,# embedding_dimension * 4 a somewhat arbitrary number, it's just the same choice as attention paper
        value_matrix_rank = 4,
        # key_query_dimension = 48, # I think this doesn't necessarily have to be the same as the embedding dimension, but it is by assumption in the GPT paper

        embedding_dropout_p = 0.1, # from GPT1 paper
        residual_dropout1_p = 0.1, # from GPT1 paper
        residual_dropout2_p = 0.1, # from GPT1 paper
        attention_dropout_p = 0.1, # from GPT1 paper


        )

class GELU(nn.Module):
    """
    I went to try and implement this by myself and then realized the paper just exhibits this approximation formula,
    so in the end I just copied this from Karpathy (who I guess in turn copied in from BERT).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# After implementing this, I realize that it was just about as much of a headache to keep track of the separate heads as it would have been to just "vectorize" them.
# Oh well.
class SelfAttentionSeparate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # naive implementation, "unvectorized"
        # this should be mathematically equivalent to the Attention is All You Need Paper, 
        # but its characterization is similar to 3blue1brown's video on attention patterns
        attention_dimension = cfg.embedding_dimension // cfg.num_attention_heads

        self.heads = []
        a = torch.arange(cfg.max_sequence_length).view(cfg.max_sequence_length, 1)
        b = torch.arange(cfg.max_sequence_length).view(1, cfg.max_sequence_length)
        self.attention_mask = (b < a)
        for head_idx in range(cfg.num_attention_heads):
            self.heads.append(nn.ModuleDict(dict(
                # I have seen implementations that cut the embedding into num_heads pieces before applying the query matrices to them (such as Karpathy's minGPT),
                # but the wording of the paper more closely suggests that the query/key matrices are what do the down-projecting to (attention dimension) space.
                # Both seem like reasonable choices, but I will pick the latter.
                query_op = nn.Linear(cfg.embedding_dimension, attention_dimension),
                key_op = nn.Linear(cfg.embedding_dimension, attention_dimension ),
                value_down = nn.Linear(cfg.embedding_dimension, cfg.value_matrix_rank),
                value_up = nn.Linear(cfg.value_matrix_rank, attention_dimension),
            )))
        self.heads = nn.ModuleList(self.heads)

        self.attention_dropout = nn.Dropout(cfg.attention_dropout_p)
        self.residual_dropout1 = nn.Dropout(cfg.residual_dropout1_p)

        pass

    def forward(self, x):
        batch_size, sequence_length, embedding_dimension = x.size()
        attention_outputs = []
        for head in self.heads:
            # these should both be of shape (batch_size, sequence_length, attention_dimension)
            keys = head.key_op(x)
            queries = head.query_op(x)
            # compute the dot products along the attention dimension, so we want a multiplication that is (sequence_length, attention_dimension) * (attention_dimension, sequence_length) at the end
            attention_pattern = torch.matmul(queries, keys.transpose(-2, -1))
            attention_pattern = attention_pattern / math.sqrt(embedding_dimension)
            attention_pattern = attention_pattern.masked_fill(self.attention_mask, float('-inf'))
            # now the last two dimensions of this are (sequence_length, sequence_length) as desired; the first index refers to the queries
            attention_pattern = F.softmax(attention_pattern, dim=-1) # softmax should be along keys, so should be last dimension
            attention_pattern = self.attention_dropout(attention_pattern)
            values = head.value_up(head.value_down(x)) # last dimension is attention_dimension
            summed_values = attention_pattern @ values
            attention_outputs.append(summed_values)
        # need to append attention outputs across heads
        output = torch.cat(attention_outputs, dim=-1)
        output = self.residual_dropout1(output)
        return output # note these are the residuals, caller must be responsible for the residual connection


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # note: GPT1 has layer norm after, GPT2 has layer norm before "sub-layers". 
        # It seems that people agree that before is better, for whatever reason.
        self.layer_norm1 = nn.LayerNorm(cfg.embedding_dimension)
        # self.attention = nn.MultiLayerAttention(cfg.embedding_dimension, cfg.num_attention_heads, dropout=cfg.attention_dropout_p, batch_first=True)
        self.attention = SelfAttentionSeparate(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.embedding_dimension)
        self.linear_layer1 = nn.Linear(cfg.embedding_dimension, cfg.linear_layer_dimension)
        self.activation = GELU()
        self.linear_layer2 = nn.Linear(cfg.linear_layer_dimension, cfg.embedding_dimension)
        self.residual_dropout2 = nn.Dropout(cfg.residual_dropout2_p)
        pass

    def forward(self, x):
        y = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x += y # res connection
        y = x
        x = self.layer_norm2(x)
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.residual_dropout2(x)
        x += y # res connection
        return x

class GPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        '''
        self.embedding_method = cfg.embedding_method
        if cfg.embedding_method is not None:
            pass
        '''
        # use learned embeddings for words and positions, rather than any preset word embedding/BPE, and any fancy sinusoid positional embedding
        self.word_embedding = nn.Embedding(cfg.num_distinct_tokens, cfg.embedding_dimension)
        self.pos_embedding = nn.Embedding(cfg.max_sequence_length, cfg.embedding_dimension)
        self.embedding_dropout = nn.Dropout(cfg.embedding_dropout_p)

        self.layer_norm_final = nn.LayerNorm(cfg.embedding_dimension)
        # self.attention = nn.MultiLayerAttention(cfg.embedding_dimension, cfg.num_attention_heads, dropout=cfg.attention_dropout_p, batch_first=True)

        self.tf_blocks = nn.ModuleList([TransformerBlock(cfg) for i in range(cfg.num_decoder_blocks)])
        self.prediction_head = nn.Linear(cfg.embedding_dimension, cfg.num_distinct_tokens)
        pass

    def forward(self, idx_seq, labels=None):

        batch_size, sequence_length = idx_seq.size()
        # if self.embedding_method 
        word_embeddings = self.word_embedding(idx_seq)
        positions = torch.arange(0, sequence_length, dtype=torch.long).unsqueeze(0) # shape (1, sequence_length); should broadcast with batch size
        position_embeddings = self.pos_embedding(positions)
        x = word_embeddings + position_embeddings
        x = self.embedding_dropout(x)
        for block in self.tf_blocks:
            x = block(x)
        x = self.layer_norm_final(x) # from GPT2
        logits = self.prediction_head(x)
        if labels is not None:
            # interpretation of these indices:
            # flatten out logits so that the logit prediction for every token in every batch is separate in the first dimension
            # flatten out labels to match first dimension, to match cross entropy API
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1) # ignore index makes hardcoded assumption that mask value for times when you're not supposed to predict is -1
        return logits, loss

    def configure_optimizers(self, train_config):
        """
        Just took this from Karpathy for now. Will reimplement it.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

