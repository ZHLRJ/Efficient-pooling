import time
import os
import torch
import torch.nn.functional as F

def save_tofile(info,file="traininfo.txt"):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, ACC: {:.4f}, '
               'Train Loss: {:.4f}, Test Loss:{:.4f}').format(
                info['current_epoch'], info['epochs'], info['t_duration'],
                info['acc'], info['train_loss'], info['test_loss'])
    if os.access(file, os.F_OK):
        with open(file, 'a') as f:
            for line in message:
                f.write(line)
            f.write('\n')
    else:
        with open(file, 'w') as f:
            for line in message:
                f.write(line)
            f.write('\n')
def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }
        # eval_info = {
        #     'train_loss': train_loss,
        #     'test_loss': test_loss,
        #     'acc': acc,
        #     'current_epoch': epoch,
        #     'epochs': epochs,
        #     't_duration': t_duration
        # }
        # # write training info to file
        # save_tofile(eval_info, file="traininfo.txt")
        # # print out info
        # if epoch % 20 == 0:
        #     print_info(eval_info)
        # #         store model
        # if epoch % (epochs // 4) == 0:
        #     torch.save(model.state_dict(), Model_PATH)
        # torch.save(model.state_dict(), Model_PATH)
        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        x = data.x.to(device)
        out = model(x)
        loss = F.l1_loss(out, x, reduction='mean')
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            pred = model(x)
            total_loss += F.l1_loss(pred, x, reduction='mean')
    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
