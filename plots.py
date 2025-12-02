'''
Read in data from metrics.csv and plot the results.
Metrics.csv: epoch,train_loss
'''
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.loadtxt('metrics.csv', delimiter=',', skiprows=1)
    print(data)
    epochs = data[:, 0]
    train_loss = data[:, 1]
    # add a subplot that display the time taken per epoch
    time = data[:, 2]
    plt.subplot(2, 1, 2)
    plt.plot(epochs, time, label='Time per Epoch', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.loglog(epochs, train_loss, label='Train Loss', color='blue')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()




if __name__ == "__main__":
    main()