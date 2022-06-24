import torch


class Training:
    def __init__(self, loss_fn, learning_rate=1e-3):
        self.__loss_fn = loss_fn
        self.__learning_rate = learning_rate

    def train_loop(self, dataloader, model, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = self.__loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def run_epochs(self):
        loss = nn.MSELoss()

        learning_rate = 1e-3
        batch_size = 64

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(colorizer.parameters(), lr=learning_rate)

        epochs = 10
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(train_dataloader, colorizer, loss_fn, optimizer)
            self.test_loop(test_dataloader, colorizer, loss_fn)
        print("Done!")
