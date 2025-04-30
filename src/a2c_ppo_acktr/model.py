import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .distributions import Bernoulli, Categorical, DiagGaussian
from .utils import init

class Normalizer:
    _STATS_FNAME = "env_stats.pickle"

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, in_size, device='cpu', dtype=torch.float):
        device='cpu'
        self.mean = torch.zeros((1, in_size), device=device, dtype=dtype)
        self.std = torch.ones((1, in_size), device=device, dtype=dtype)
        self.eps = 1e-12 if dtype == torch.double else 1e-5
        self.device = device
        self.count = self.eps

    def update_stats(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(data.device)
        data = data.to('cpu')
        batch_mean = data.mean(0, keepdim=True)
        batch_var = data.var(0, keepdim=True)
        batch_count = data.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = torch.square(self.std) * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.std = torch.sqrt(new_var)
        self.count = new_count

    def normalize(self, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        return (val - self.mean.to(val.device)) / std.to(val.device)

    def denormalize(self, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        return std * val.to(val.device) + self.mean.to(val.device)
 

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].
        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, args, obs_shape, action_space, use_attention, device, block_size, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
                self.base = base(obs_shape[0], **base_kwargs)
            elif len(obs_shape) == 1:
                if use_attention:
                    cfg = TransformersArchitecture.get_default_config(args)
                    cfg.model_type = 'gpt-nano'
                    cfg.block_size = block_size
                    cfg.input_size = int(obs_shape[0] / cfg.block_size)
                    cfg.device = device
                    cfg.act_size = action_space.n
                    self.base = TransformersArchitecture(cfg)
                else:
                    base= MLPBase
                    self.base = base(obs_shape[0], **base_kwargs)
            else:
                raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def _to(self, gpu_id):
        self.base.to(f"cuda:{gpu_id}")
        self.dist.to(f"cuda:{gpu_id}")

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
            dist = self.dist(actor_features)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0).to('cuda:0'), (hxs.to('cuda:0') * masks.to('cuda:0')).unsqueeze(0).to('cuda:0'))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        print('shape input: ', num_inputs)

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), 
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), 
            nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), 
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), 
            nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if torch.isnan(inputs).any():
            print("INPUTS: ", inputs)
            print("Input data contains NaN values!")

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
    
# Additional part for attention model
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class DecoderBlock(nn.Module):
    """An Decoder block of a Transformer."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = nn.Sequential(nn.Linear(config.n_embd, config.n_embd * 4),
        NewGELU(),
        nn.Linear(config.n_embd * 4, config.n_embd),
        nn.Dropout(config.resid_pdrop))
    
    def forward(self, x):
        try:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlpf(self.ln_2(x))
        except Exception as e:
            print('Input x:', x)
            print(e)
            print(x.shape)
            raise e
        return x

class TransformersArchitecture(nn.Module):
    """This class implements the Transformer architecture."""

    @staticmethod
    def get_default_config(args):
        C = CfgNode()
        C.model_type='gpt'
        # either model_type or n_layer, n_head, n_embd should be specified
        C.n_layer=None
        C.n_head=None
        C.n_embd=None
        # these option must be filled externally
        C.vocab_size=None
        C.block_size=None
        C.embd_pdrop=0.0
        C.resid_pdrop=0.0
        C.attn_pdrop=0.0
        if args is not None:
            C.ac_representation = args.ac_representation
        return C
    
    def __init__(self, config) -> None:
        super().__init__()
        assert config.block_size is not None
        # config.block_size += 1 # This is because for each block of sequence, we want to add one last block as the place holder for the action
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert ((type_given and not params_given) or (not type_given and params_given)), "Either model_type or n_layer, n_head, n_embd should be specified"
        if type_given:
            # take the config from type_given
            config.merge_from_dict({
                # GPT-1
                'gpt1': dict(n_layer=12, n_head=12, n_embd=768),

                # GPT-2
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),

                # custom
                'gpt-nano': dict(n_layer=4, n_head=4, n_embd=128),
            }[config.model_type])
    
            print("config input: ", config.input_size)

            # block here might mean sequence block, not a vertical one

            self.wte = nn.Linear(config.input_size, config.n_embd) # map input to embedding space ()
            self.wpe = nn.Embedding(config.block_size, config.n_embd) # positional encoding
            # self.wae = nn.Embedding(config.act_size, config.n_embd) # action encoding
            self.drop = nn.Dropout(config.embd_pdrop)
            self.h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
            self.ln_f = nn.LayerNorm(config.n_embd)
            if config.ac_representation == 'concat':
                self.base = MLPBase(config.n_embd * config.block_size, recurrent=False, hidden_size=1024)
            elif config.ac_representation == 'mean' or config.ac_representation == 'last':
                self.base = MLPBase(config.n_embd, recurrent=False, hidden_size=1024)
            self._hidden_size = 1024
            self._recurrent = False
            # NOTE: changed this
            self.normalizer = Normalizer(config.block_size * config.input_size, device=config.device)

            # init all weights, and apply a special init to the residual projections, per GPT-2 paper
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

            # report number of parameters
            n_params = sum(p.numel() for p in self.parameters())
            print('Number of parameters: %.2fM' % (n_params/1e6,))

            # get config
            self.config = config

    def _init_weights(self, module):
        """ Initialize the weight matrices """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Initialize a pretrained GPT model by copying over the weights
        from a huggingface/ transformers checkpoint.
        """

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        config = cls.get_default_config(None)
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocab
        config.block_size = 1024 # openai's model block size
        model = TransformersArchitecture(config)
        sd = model.state_dict()

        # init a huggingface/ transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear for attention
        # this means that we have to transpose these weights when we import them

        assert len(keys) == len(sd)
        for k in keys():
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
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

    def forward(self, idx, rnn_hxs, masks):
        """
        In this fucntion, we notice that the input for transformers is prepared, by:
        1. Divide the initial state into sequence of tokens & action (the action is one special token at the end of the sentence)
        2. Add positional embeddings for each token
        3. Push the token through nn.Embeddings to get the embedding representations, then concat the sentence embeddings with the action embedding
        4. Push the sequence of sentence embeddings through transfoermers with n_layers blocks to get the transformer matrices
        5. Go though another linear layer # TODO: check what is this for
        6. Flatten the state  representations again and push them as outputs
          """
        # idx: B x 4 x D
        # idx is actually the state that RL agent sees. In this part we want to transform it into the right shape of input for transformers
        device = idx.device

        # B x D: take the representations for every (4) actions
        # print("===============")
        # removing action from state
        # action = idx[:, -1].long()

        # TODO: check this later - what is idx
        # idx = self.normalizer.normalize(idx)[:, :-1] 
        # idx = idx[:, :-1]# NOTE: why -1? because the last dimension we are using to represent action, thus we have to chop it off before dividing state into transformers input
        idx = idx.reshape(idx.shape[0], self.block_size, -1)
        b, t, _ = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1,t)

        # print("Idx shape after chopped off: ", idx.shape)
        # B x 4 x D1
        tok_emb = self.wte(idx)

        # B x 1 x D1
        # act_emb = self.wae(action).unsqueeze(1) # action embeddings of shape (b,1,n_embd)

        # B x 4 x D1
        # tok_emb = torch.cat([tok_emb, act_emb], dim=1)
        # print("tok_emb shape: ", tok_emb.shape)
        pos_emb = self.wpe(pos) # position embeddings of shape (1,t,n_embd)
        # print("pos_emb shape: ", pos_emb.shape)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)
        # print("After block: ", x.shape)

        # B x 4 x D1
        x = self.ln_f(x)
        # print("After ln_f: ", x.shape)

        # B x (4 x D1)
        if self.config.ac_representation == 'concat':
            x = x.reshape(x.shape[0], -1)
        elif self.config.ac_representation == 'mean':
            x = x.mean(dim=1)
        elif self.config.ac_representation == 'last':
            x = x[:, -1, :]
        # print("Shape of x: ", x.shape)
        # this reshapes the x input, as we want to feed the base the flattened input.
        return self.base(x, rnn_hxs, masks)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size
