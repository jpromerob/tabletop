import socket
import pyNN.spiNNaker as p
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import time
import math
import numpy as np
import pdb
from numpy import genfromtxt


def plot_in_v_out(i_indexes, v, o_indexes, label, xlim):

    nsll= 0.4
    
    rv = 4    
    ri = (1+1)*nsll
    ro =(1+1)*nsll
    
    height = (ri+rv+ro)+2    
    width = 15
    
    # Plot spikes and voltages
    fig, axs = plt.subplots(3, figsize=(width,height), gridspec_kw={'height_ratios': [ri, rv, ro]})
    fig.tight_layout(pad=5.0)
    fig.suptitle(label)

    axs[0].eventplot(i_indexes, linewidths=2, colors='k', linelengths=nsll) # Plot the timesteps where the neuron spiked
    axs[0].set_xlabel("Time [ms]")
    axs[0].set_ylabel("Input")
    axs[0].set_xticks(np.arange(0, xlim+1, 10))
    axs[0].set_yticks(np.arange(0, 3, 1))

    axs[1].plot(v, linewidth=2)
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel("Voltage [ms]")
    axs[1].set_xlim((0,xlim))
    axs[1].set_ylim((-66,-59))

    axs[2].eventplot(o_indexes, linewidths=2, colors='k', linelengths=nsll) # Plot the timesteps where the neuron spiked
    axs[2].set_xlabel("Time [ms]")
    axs[2].set_ylabel("Output")
    axs[2].set_xticks(np.arange(0, xlim+1, 10))
    axs[2].set_yticks(np.arange(0, 3, 1))
    
    plt.savefig(f"images/{label}.png")
    
def plot_all(data_list, name):
    # pdb.set_trace()
    nsll= 0.4
    
    rv = 4    
    ri = (1+1)*nsll
    ro =(1+1)*nsll
    
    height = (ri+rv+ro)+2    
    width = 15
    
    # Plot spikes and voltages
    fig, axs = plt.subplots(2, figsize=(width,height), gridspec_kw={'height_ratios': [ri, rv]})
    fig.tight_layout(pad=5.0)
    fig.suptitle("Checking neuron parameters ... ")
    
    # pdb.set_trace()
    plot_counter = 0
    for (label, i_indexes, v_array, o_indexes, xlim) in data_list:

        if plot_counter == 0:
            axs[0].eventplot(i_indexes, linewidths=2, colors='k', linelengths=nsll) # Plot the timesteps where the neuron spiked
            axs[0].set_xlabel("Time [ms]")
            axs[0].set_ylabel("Input")
            axs[0].set_yticks(np.arange(0, 3, 1))
            axs[0].set_xlim((0,xlim))
            plot_counter+=1

        axs[1].plot(v_array, linewidth=2, label=label)
        axs[1].set_xlabel("Time [ms]")
        axs[1].set_ylabel("Voltage [ms]")
        axs[1].set_xlim((0,xlim))
        axs[1].set_ylim((-66,-59))


    axs[1].grid(which='minor', axis='x', linestyle='--', alpha=0.5)
    axs[1].minorticks_on()
    axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[1].legend()

    plt.savefig(f"images/{name}.png")
    
    
def generate_trains(period):

    i_spikes = [0,0,0]
    nb_pts = 9*period
    for i in range(nb_pts):
        if i%period == 0:
            i_spikes.append(1)
        else:
            i_spikes.append(0)
    np.savetxt("input_train.csv", i_spikes, delimiter=",")


def set_parameters(dt, w, tm, ts):
    dt = 1

    cell_params = {'tau_m': tm/math.log(2),
                   'tau_syn_E': ts,
                   'tau_syn_I': ts,
                   'v_rest': -65.0,
                   'v_reset': -65.0,
                   'v_thresh': -60.0,
                   'tau_refrac': 0.0, # 0.1 originally
                   'cm': 1,
                   'i_offset': 0.0
                   }
    
    return dt, w, cell_params


def do_stuff(dt, w, tm, ts):
    
    label = f"w{w}_tm_{tm}_ts_{round(ts,2)}"   
    print(label)
    dt, w, cell_params = set_parameters(dt, w, tm, ts)

    #Fetch available trains of spikes in path
    f_path = "input_train.csv"

    #SpiNNaker Setup
    node_id = p.setup(timestep=1)     
    p.set_number_of_neurons_per_core(p.IF_curr_exp, 100) #  100 neurons per core

    i_spikes = genfromtxt(f_path, delimiter=',')
    i_indexes = np.where(i_spikes>0)
    nb_steps = len(i_spikes)

    # Populations
    celltype = p.IF_curr_exp
    cells_l1 = p.Population(1, celltype(**cell_params), label="Layer_1")

    spike_train_1 = p.SpikeSourceArray(spike_times=(i_indexes))
    cells_l0 = p.Population(1,spike_train_1)

    # Connectivity
    cell_conn = p.AllToAllConnector()
    connections = { 'i1l1': p.Projection(cells_l0, cells_l1, cell_conn,
                            receptor_type='excitatory',
                            synapse_type=p.StaticSynapse(weight=w, delay=0))}

    # Setup recording 
    cells_l1.record(["v","spikes"])
    cells_l0.record(["spikes"])


    # Run simulation 
    p.run(nb_steps)


    # Print results to file 

    l1_voltage = cells_l1.get_data("v")
    l1_spikes = cells_l1.get_data("spikes")
    in_spikes = cells_l0.get_data("spikes")

    # Finished with simulator 
    p.end()


    time.sleep(10)
    v_array = np.array(l1_voltage.segments[0].filter(name="v")[0]).reshape(-1)

    i_indexes = np.asarray(in_spikes.segments[0].spiketrains[0])
    o_indexes = np.asarray(l1_spikes.segments[0].spiketrains[0])

    o_spikes = np.zeros(nb_steps)
    for i in o_indexes.astype(int):
        o_spikes[i] = 1

    #np.savetxt("summary/spinnaker/" + "voltage_" + sfn, v_array, delimiter=",")
    #np.savetxt("summary/spinnaker/" + "output_" + sfn, o_spikes, delimiter=",")

    xlim = len(i_spikes)
    return label, i_indexes, v_array, o_indexes, xlim

















if __name__ == '__main__':

    
    dt = 1

    period_list =np.array([4,8])
    w_array = np.array([0.2,0.4])
    tm_array = np.linspace(1,9,3)
    ts_array = np.array([0.1,0.5,1.0,2.0])


    # pdb.set_trace()

    os.system("rm -rf reports/")
    os.system(f"rig-power 172.16.223.0")

    for period in period_list:
        generate_trains(period)
        for w in w_array:
            for tm in tm_array:
                data_list = []
                name = f"p_{period}_w_{w}_tm_{tm}_variable_ts"
                for ts in ts_array:
                    label, i_indexes, v_array, o_indexes, xlim = do_stuff(dt, w, tm, ts*tm)
                    data_list.append((label, i_indexes, v_array, o_indexes, xlim))
                plot_all(data_list, name)    