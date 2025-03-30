import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    # training time
    single_mean, single_std = 29.19621169567108, 0.09175336360931396
    device0_mean, device0_std =  17.11961579322815, 0.405240535736084
    device1_mean, device1_std =  16.835357666015625, 0.25300145149230957
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_training_time.png')
    
    # tokens per sec
    single_mean, single_std = 124234.309205573, 153.230292656611
    device0_mean, device0_std =  122917.26783198275, 1259.7318543649963
    device1_mean, device1_std =  122520.75055418984, 1280.51564048187
    double_mean = device0_mean + device1_mean
    double_std = (device0_std ** 2 + device1_std ** 2) ** 0.5
    plot([double_mean, single_mean],
        [double_std, single_std],
        ['Data Parallel', 'Single GPU'],
        'ddp_tokens_per_sec.png')

    # pp_mean, pp_std = None, None
    # mp_mean, mp_std = None, None
    # plot([pp_mean, mp_mean],
    #     [pp_std, mp_std],
    #     ['Pipeline Parallel', 'Model Parallel'],
    #     'pp_vs_mp.png')