import sys
sys.path.append("/home/s223540177/dai/RLforLLM/lm_evaluation_harness")

from tqdm import tqdm

from lm_eval.__main__ import run_eval, parse_eval_args
from src.sample_memory import Memory
from src.data_utils import custom_load_dataset
from lm_evaluation_harness.lm_eval.api.task import ConfigurableTask
from lm_evaluation_harness.lm_eval.api.samplers import ContextSampler

"""
NOTE: To adapt to a new set, fix 'dataset_promptsource_name' in task.py
"""

def setup_memory(args, n_actions, dataset_promptsource_name):
    from promptsource.templates import DatasetTemplates

    n_states = 16

    
    order_memory = Memory(args, n_states, n_actions)

    return order_memory

def main():
    args = parse_eval_args()

    if args.baseline == 'memory':
        n_actions = args.n_actions 
        if n_actions == 24:
            total_episodes = 65
        elif n_actions == 360:
            total_episodes = 625
        elif n_actions == 1680:
            total_episodes = 65
        eval_interval = 10
    elif args.baseline == 'all':
        n_actions = args.n_actions
        total_episodes = 1000
        eval_interval = 100
    elif args.baseline in ['ascending', 'descending']:
        n_actions = args.n_actions
        total_episodes = 1
        eval_interval = 1

    params = {
            "dataset": args.dataset_promptsource_name,
            }

    acc_max = 0.0

    # NOTE: run this once for every new task
    _, _, _, _, _, _ = custom_load_dataset(params)

    order_memory = setup_memory(args, n_actions, args.dataset_promptsource_name)

    if args.dataset_promptsource_name == 'anli':
        task_name_train = f"{params['dataset']}_r1_train"
        task_name_eval = f"{params['dataset']}_r1_eval"
    else:
        if args.baseline == 'ascending':
            task_name_train = f"{params['dataset']}_train_asc"
            task_name_dev = f"{params['dataset']}_dev_asc"
            task_name_eval = f"{params['dataset']}_eval_asc"
        elif args.baseline == 'descending':
            task_name_train = f"{params['dataset']}_train_des"
            task_name_dev = f"{params['dataset']}_dev_des"
            task_name_eval = f"{params['dataset']}_eval_des"
        else:
            task_name_train = f"{params['dataset']}_train"
            task_name_dev = f"{params['dataset']}_dev"
            task_name_eval = f"{params['dataset']}_eval"

    if args.baseline in ['ascending', 'descending']:
        print("EVALUATION")
        _, _, _ = run_eval(order_memory, args, task_name_eval, 0, evaluate=True)
    elif args.baseline == 'memory':
        for i in tqdm(range(total_episodes), desc="Training"):
            # print("Order Memory: ", order_memory.values)
            # print("======")
            results, task_manager, _ = run_eval(order_memory, args, task_name_train, i, evaluate=False)
            # access the task
            task_object = task_manager.get_task()

            # pass the current episode number to task_object

            # access the context sampler
            sampler = task_object.sampler
            if args.baseline =='all':
                task_object.set_config("total_episodes", 1000)

            # update the sampler

            # print("Results: ", results[task_name_train][-1])
            sampler.update_order_memory(results[task_name_train])


            if i > 0 and i % eval_interval == 0:
                print("DEV EVALUATION")
                _, _, result = run_eval(order_memory, args, task_name_dev, i, evaluate=True)

                # print("Current memory: ", order_memory.values)
                # quit()
                this_acc = float(result[0][5])
                if this_acc >= acc_max:
                    print(f" CURRENT DEV ACC {this_acc} >= MAX DEV ACC {acc_max} - > EVALUATION")
                    acc_max = this_acc
                    _, _, final_results = run_eval(order_memory, args, task_name_eval, i, evaluate=True)


        print("FINAL ACC: ", final_results[0][5])
if __name__ == "__main__":
    main()