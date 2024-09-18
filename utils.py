import matplotlib.pyplot as plt

# 绘制Train-Valid loss图
def show_plot(train_losses_history, valid_losses_history, loss_pic_filename):
    plt.plot(train_losses_history, label = 'Training Loss', color = 'blue')
    plt.plot(valid_losses_history, label = 'Validation Loss', color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(loss_pic_filename)