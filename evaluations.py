import os.path
import sys
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import json
import numpy as np


mypath = "./results/"
jsonfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
jsonfiles.sort()

def main(file_name):
    """
    file_name
    Args:
        file_name:

    Returns:

    """
    with open(os.path.join(mypath, file_name), 'r') as f:
        stat_data = json.load(f)

    ##get the info by round
    round_data = stat_data.get("by_round", {})
    round_names = list(round_data.keys())
    round_names.sort()
    n_rounds = len(round_names)
    if n_rounds == 0:
        return
    stats_by_round = {"coins": [], "kills": [], "suicides": []}
    for r in round_names:
        for key in stats_by_round:
            if key in round_data[r]:
                stats_by_round[key].append(round_data[r][key])
            else:
                stats_by_round[key].append(0)
    ## get the info by agent
    agent_data = stat_data.get("by_agent", {})
    agents = {}
    label = ["coins", "kills", "invalid", "score", "suicides", "bombs", "crates", "moves"]
    for key in agent_data:
        agents[key] = agents.get(key, {"name": key})
        for l in label:
            agents[key][l] = agent_data[key].get(l, 0)
    print(agents)
    ## plot
    fig, ax = plt.subplots(4, figsize=(7.2, 12), sharex=False)
    # by round
    ax[0].plot(stats_by_round["coins"], label="coins")
    ax[0].set_title('stats by round')
    ax[0].set_ylabel('coins')

    ax[1].plot(stats_by_round["kills"], label="kills")
    ax[1].set_title('stats by round')
    ax[1].set_ylabel('kills')

    ax[2].plot(stats_by_round["suicides"], label="suicides")
    ax[2].set_title('stats by round')
    ax[2].set_ylabel('suicides')
    # by agent
    label1 = ["coins", "kills", "invalid", "score", "suicides"]
    v_list = []
    for agent in agents:
        cur = [agents[agent][key] for key in label1]
        cur.append(agents[agent]["name"])
        v_list.append(cur)
    assert len(v_list) == 4
    #print(v_list)

    x = np.arange(len(label1))  # the label locations
    width = 0.10  # the width of the bars
    #print(x - width, v_list[0][:-2], v_list)
    ax[3].bar(x - width * 1.5 , v_list[0][:-1], width, label=f'{v_list[0][-1]}')
    ax[3].bar(x - width / 2, v_list[1][:-1], width, label=f'{v_list[1][-1]}')
    ax[3].bar(x + width / 2, v_list[2][:-1], width, label=f'{v_list[2][-1]}')
    ax[3].bar(x + width * 1.5, v_list[3][:-1], width, label=f'{v_list[3][-1]}')
    ax[3].set_ylabel('Scores')
    ax[3].set_title('Scores by agent')
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(label1)
    ax[3].legend(loc='upper left')


    """
    label2 = ["bombs", "crates"]
    v_list0 = [agent_0[key] for key in label2]
    v_list1 = [agent_1[key] for key in label2]
    x = np.arange(len(label2))  # the label locations
    width = 0.35  # the width of the bars
    ax[4].bar(x - width / 2, v_list0, width, label=f'{agent_0["name"]}')
    ax[4].bar(x + width / 2, v_list1, width, label=f'{agent_1["name"]}')
    ax[4].set_ylabel('Scores')
    ax[4].set_title('Scores by agent')
    ax[4].set_xticks(x)
    ax[4].set_xticklabels(label2)
    ax[4].legend()
    """

    fig.tight_layout()
    plt.savefig(f'./plot-{file_name[0:-5]}.pdf')
    plt.close('all')



if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].find("json") >0 else jsonfiles[-1]
    #agent_name = sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1].find("json") < 0 else ""
    print(file_name)
    main(file_name)





