import torch

def get_training_args():
    import argparse
    parser = argparse.ArgumentParser(description='Edit Prompt')

    # OVERALL FRAMEWORK
    parser.add_argument('--max_num_edits', type=int, default=1, help='Maximum number of edits allowed')
    parser.add_argument('--task_format', type=str, default='classification', choices=['classification', 'qa', 'test'], help='The format of the task')
    parser.add_argument('--reward_model_name', default='roberta-large', type=str, choices=['mistralai/Mistral-7B-Instruct-v0.2', 'roberta-large', 'llama-v2-7B-chat'], help='The name of the model to use for downstream task')
    parser.add_argument('--senemb_model_name', default='mixedbread-ai/mxbai-embed-large-v1', type=str, choices=['Salesforce/SFR-Embedding-Mistral', 'mixedbread-ai/mxbai-embed-large-v1', 'Alibaba-NLP/gte-large-en-v1.5'], help='The name of the model to use for sentence embedding')
    parser.add_argument('--n_pool_exs', type=int, default=16)
    parser.add_argument('--extra_exs', action='store_true', default=False, help='whether to use extra examples or not')
    parser.add_argument('--use_bm25', action='store_true', default=False, help='whether to use bm25 or not')

    parser.add_argument('--prompt_format', type=str, default='concat', choices=['concat', 'newline'], help='The format of the prompt')
    parser.add_argument('--elements', default='all', help='The elements to use for the framework')
    parser.add_argument('--reward_type', type=str, default='prob', choices=['prob', 'acc'], help='The type of reward to use for RL training')
    parser.add_argument('--epsilon', type=int, default=0.1, help='The epsilon value for epsilon greedy')
    parser.add_argument('--ac_representation', type=str, default='mean', choices=['mean', 'last', 'all'], help='The type of input to use for the model')
    parser.add_argument('--example_sample_type', type=str, default='example_selection', choices=['example_selection', 'permutation_selection', 'narrowed_example_selection'], help='The type of example sampling to use')
    parser.add_argument('--n_slots', type=int, default=4, help='Number of in-context examples in each prompt')
    parser.add_argument('--num_actors', default=8, type=int, metavar='N', help='Number of actors.')
    parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_const', const=True, default=False,
                        help='Whether to load ckpt')
    parser.add_argument('--inspect', action='store_true', default=False, help='whether to inspect memory time or not')
    parser.add_argument('--imbalanced_labels', action='store_true', default=False, help='whether to use imbalanced labels or not')
    parser.add_argument('--n_actions', type=int, default=24)
    parser.add_argument('--verbalizer_index', type=int)

    parser.add_argument('--furthest_neighbor', action='store_true', default=False, help='whether to use furthest neighbors or not')
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['cosine', 'euclidean'], help='The distance metric to use for example selection')

    parser.add_argument('--visualize', action='store_true', default=False, help='whether to visualize or not')

    parser.add_argument('--heuristic_order', type=str, default='descending', choices=['ascending', 'descending'], help='The order to use for heuristic')
    
    parser.add_argument('--baseline', type=str, default='memory', choices=['memory', 'memory_qa', 'all', 'random_random', 'random_topk', 'memory_random', 'heuristic', 'memory_seperate'], help='whether to optimized with memory or all permutations')
    parser.add_argument('--memory_k', type=int, default=10)
    parser.add_argument('--eval_data', type=str, default='valid', choices=['test', 'valid'])
    parser.add_argument('--small_pool', action='store_true', default=False, help='whether to use small pool or use whole dataset to create examples')
    parser.add_argument('--use_verbalizer', action='store_true', default=False, help='whether to use verbalizer')

    parser.add_argument('--write_type', type=str, default='max', choices=['max', 'mean'], help='The type of write to use for the model')
    parser.add_argument('--debug_mode', action='store_true', default=False, help='whether to use debug mode or not')
    parser.add_argument('--type_embd', type=str, default='llama', choices=['llama', 'senemb', 'mistral'])



    parser.add_argument('--distance_threshold', type=float, default=0.1, help='The distance threshold for example selection')

    # ACTION LLM
    # parser.add_argument('--model_path', default='/weka/Projects/local_llms/model_weights/llama-v2-13B-chat/', help='The path to the model used for text generation')
    parser.add_argument('--model_path', default="mistralai/Mixtral-8x7B-Instruct-v0.1", help='The path to the model used for text generation')
    parser.add_argument('--load8bit', action='store_true', default=False, help='Whether to load the 8-bit model or not')
    # CHANGED THIS TO 16bit
    parser.add_argument('--load4bit', action='store_true', default=True, help='Whether to load the 4-bit model or not')

    parser.add_argument('--max_seq_len', type=int, default=2048, help='The maximum sequence length for input prompts')
    parser.add_argument('--max_gen_len', type=int, default=256, help='The maximum length of generated sequences. If None, it will be set to the model\'s max sequence length')

    parser.add_argument('--n_candidates', type=int, default=3, help='The number of candidates to generate for each input prompt')
    parser.add_argument('--tokenizer_path', default='/home/s223540177/dai/llama/tokenizer.model', help='The path to the tokenizer model used for text encoding/decoding')
    parser.add_argument('--temperature', type=float, default=0.6, help='The temperature value for controlling randomness in generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='The top-p sampling parameter for controlling diversity in generation')
    parser.add_argument('--max_batch_size', type=int, default=8, help='The maximum batch size for generating sequences')

    max_tokens_dict = {
        'roberta-large': 1024
    }

    
    # REWARD LLM
    parser.add_argument('--dataset', default='glue/sst2', type=str, help='The name of the model to use for text generation')
    parser.add_argument('--state_space', default='tokens', help='The way to represent state')
    parser.add_argument('--obs_size', default=4096, help='The hidden size of the model')
    parser.add_argument('--num_examples', default=2, help='NUM_IN_CONTEXT = num_examples * 2')
    parser.add_argument('--batch_size', default=64, help='Batch size')
    parser.add_argument('--use_attention', action='store_true', default=False, help='Whether to use attention or not')


    # MEMORY
    parser.add_argument('--use_memory', action='store_true', default=False, help='whether to use memory or not')
    parser.add_argument('--top_k', type=int, default=None, help='whether to use top_k to sample from memory in eval mode or not')
    parser.add_argument('--keep_const_examples', action='store_true', default=False, help='whether to keep constant examples or not')
    parser.add_argument('--memory_store_mode', type=str, default='max', help='whether to use mean or max to store in memory')

    parser.add_argument('--sample_type', type=str, default='weights', help='weights | epsilon_greedy')
    
    # ARGS
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')

    parser.add_argument('--wandb', action='store_true', default=False, help='use wandb for logging')
    parser.add_argument('--num-iterations', type=int, default=2, help='number of random iterations run to get the average')

    parser.add_argument('--mode', default='rl', help='mode to run model| rl, random, fixed')
    parser.add_argument('--save-data', default=False, help='whether to save the data or not')
    parser.add_argument('--parser', default='second', help='indicates which line is being parsed')

    parser.add_argument('--verbalize', action='store_true', default=True, help='whether to verbalize or not')

    parser.add_argument('--normalize_obs', action='store_true', default=True, help='whether to normalize observation or not')
    parser.add_argument('--normalize_rew', action='store_true', default=True, help='whether to normalize reward or not')

    parser.add_argument(
            '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=6e-4, help='learning rate (default: 3e-4)') # previous was 6e-4
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99, # previous was 0.999
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num_processes',
        type=int,
        default=64,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=16,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    # should set this to 1000 or 100
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1000000,
        help='number of environment steps to train (default: 1e6)')
    parser.add_argument(
        '--env-name',
        default='PromptEnv',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/home/s223540177/dai/RLforLLM/agent_log',
        help='directory to save agent logs (default: ')
    parser.add_argument(
        '--save-dir',
        default='/home/s223540177/dai/RLforLLM/trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args

