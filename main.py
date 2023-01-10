import sys
import needle as ndl
import numpy as np

from apps.linear_vit import LinearVit

np.random.seed(0)

sys.path.append("./python")
sys.path.append("./apps")

# device = ndl.cpu()
device = ndl.cuda()


def epoch_general(
    dataloader, model, loss_fn=ndl.nn.SoftmaxLoss(), opt=None, device=None
):
    if opt:
        model.train()
    else:
        model.eval()
    correct, total_loss = 0, 0
    cnt = 0
    for i, batch in enumerate(dataloader):
        if opt:
            opt.reset_grad()
        X, y = batch
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        cnt += X.shape[0]
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        if opt:
            loss.backward()
            opt.step()
        if opt and i % 10 == 0:
            print(
                f"After update {i} times, the loss is {total_loss / cnt}, the accuracy is {correct / cnt}."
            )

    return correct / cnt, total_loss / cnt


train_data = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
test_data = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=False)
print(f"number of train dataset: {train_data.n}")
print(f"number of test dataset: {test_data.n}")
train_dataloader = ndl.data.DataLoader(
    dataset=train_data, batch_size=100, shuffle=False
)
test_dataloader = ndl.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)
model = LinearVit(device=device, dtype="float32")
loss_fn = ndl.nn.SoftmaxLoss()
opt = ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
epochs = 10
for i in range(epochs):
    print(f"Start epoch {i}")
    train_acc, train_loss = epoch_general(
        dataloader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        opt=opt,
        device=device,
    )

    test_acc, test_loss = epoch_general(
        dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        opt=None,
        device=device,
    )

    print(
        f"After training {i} epochs, the loss is {test_loss}, the accuracy is {test_acc}."
    )