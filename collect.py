import numpy as np
import matplotlib.pyplot as plt
import csv

def collect():
    samples=[]
    samples.append(np.loadtxt('rewards.dat'))
    samples.append(np.loadtxt('rewards_perturbation.dat'))
    mean=np.zeros(2)
    std=np.zeros(2)
    for i in range(2):
        mean[i]=np.mean(samples[i][:,1])
        std[i]=np.std(samples[i][:,1])
    plt.figure(0)
    print('mean', mean)
    print('std', std)
    plt.plot(samples[0][800:,0], samples[0][800:,1], label='origin')
    plt.plot(samples[1][800:,0], samples[1][800:,1], label='perturbed')
    plt.xlabel('$Episodes$')
    plt.ylabel('rewards')
    plt.legend()
    plt.show()

def collect_frequency():
    samples=[]
    samples.append(np.loadtxt('./dy_4/samples.dat'))
    samples.append(np.loadtxt('./dy_3/samples.dat'))
    samples.append(np.loadtxt('./dy_2/samples.dat'))
    samples.append(np.loadtxt('./dy_1/samples.dat'))
    samples.append(np.loadtxt('./dy0/samples.dat'))
    samples.append(np.loadtxt('./dy1/samples.dat'))
    samples.append(np.loadtxt('./dy2/samples.dat'))
    samples.append(np.loadtxt('./dy3/samples.dat'))
    samples.append(np.loadtxt('./dy4/samples.dat'))
    mean=np.zeros(9)
    std=np.zeros(9)
    for i in range(9):
        counts=np.sum(samples[i][:, 1:], axis=1)
        for j in range(counts.shape[0]):
            if counts[j]!=0:
                samples[i][j,1:]/=counts[j]

    plt.figure(0)
    plt.plot(samples[0][:,0], samples[0][:,1],'o-', label='$\Delta y=-4$')
    plt.plot(samples[0][:,0], samples[1][:,1], label='$\Delta y=-3$')
    plt.plot(samples[0][:,0], samples[2][:,1], label='$\Delta y=-2$')
    plt.plot(samples[0][:,0], samples[3][:,1], label='$\Delta y=-1$')
    plt.plot(samples[0][:,0], samples[4][:,1], label='$\Delta y=0$')
    plt.plot(samples[0][:,0], samples[5][:,1], label='$\Delta y=1$')
    plt.plot(samples[0][:,0], samples[6][:,1], label='$\Delta y=2$')
    plt.plot(samples[0][:,0], samples[7][:,1], label='$\Delta y=3$')
    plt.plot(samples[0][:,0], samples[8][:,1], label='$\Delta y=4$')
    plt.xlabel('$v_y^b$')
    plt.ylabel('ratio')
    plt.legend()

    plt.figure(1)
    plt.plot(samples[0][:,0], samples[0][:,2],'o-', label='$\Delta y=-4$')
    plt.plot(samples[0][:,0], samples[1][:,2], label='$\Delta y=-3$')
    plt.plot(samples[0][:,0], samples[2][:,2], label='$\Delta y=-2$')
    plt.plot(samples[0][:,0], samples[3][:,2], label='$\Delta y=-1$')
    plt.plot(samples[0][:,0], samples[4][:,2], label='$\Delta y=0$')
    plt.plot(samples[0][:,0], samples[5][:,2], label='$\Delta y=1$')
    plt.plot(samples[0][:,0], samples[6][:,2], label='$\Delta y=2$')
    plt.plot(samples[0][:,0], samples[7][:,2], label='$\Delta y=3$')
    plt.plot(samples[0][:,0], samples[8][:,2], label='$\Delta y=4$')
    plt.xlabel('$v_y^b$')
    plt.ylabel('ratio')
    plt.legend()

    plt.figure(2)
    plt.plot(samples[0][:,0], samples[0][:,3],'o-', label='$\Delta y=-4$')
    plt.plot(samples[0][:,0], samples[1][:,3], label='$\Delta y=-3$')
    plt.plot(samples[0][:,0], samples[2][:,3], label='$\Delta y=-2$')
    plt.plot(samples[0][:,0], samples[3][:,3], label='$\Delta y=-1$')
    plt.plot(samples[0][:,0], samples[4][:,3], label='$\Delta y=0$')
    plt.plot(samples[0][:,0], samples[5][:,3], label='$\Delta y=1$')
    plt.plot(samples[0][:,0], samples[6][:,3], label='$\Delta y=2$')
    plt.plot(samples[0][:,0], samples[7][:,3], label='$\Delta y=3$')
    plt.plot(samples[0][:,0], samples[8][:,3], label='$\Delta y=4$')
    plt.xlabel('$v_y^b$')
    plt.ylabel('ratio')
    plt.legend()

    plt.show()

collect()
#collect_frequency()
