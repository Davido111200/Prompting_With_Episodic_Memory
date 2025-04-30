import pickle

from utils import plot_mean_and_variance_from_pkls

accuracy_list_noedit_path = ["/home/s223540177/dai/RLforLLM/src/results/noedit_run_1.pkl" ,\
                            "/home/s223540177/dai/RLforLLM/src/results/noedit_run_2.pkl", \
                            "/home/s223540177/dai/RLforLLM/src/results/noedit_run_3.pkl"]
save_path = "/home/s223540177/dai/RLforLLM/src/figs/"

plot_mean_and_variance_from_pkls(accuracy_list_noedit_path, save_path, "NoEdit1", care=True)

