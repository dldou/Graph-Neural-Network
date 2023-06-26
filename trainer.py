import torch


class Trainer:

    def __init__(self, device):
        print("Starting trainer init...")
        self.device = device
        print("Trainer init done.\n")

    def train_step(self, model, adj_mat,
                   dataloader,
                   optimizer, criterion):

        running_loss = 0.0
        model.train()

        for images, targets in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            # Predictions
            targets_hat = model(images, adj_mat)
            #Loss + backprop
            loss = criterion(targets_hat, targets)
            loss.backward()
            optimizer.step()

            # item() method detach automatically from the graph
            running_loss += loss.item()
        return running_loss / len(dataloader)

    def valid_step(self, model, adj_mat, valid_loader, criterion):

        model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        n_predictions = 0

        model.eval()
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(
                    self.device), targets.to(self.device)

                # Inferences
                targets_hat = model(images, adj_mat)

                # loss
                loss = criterion(targets_hat, targets)
                running_loss += loss.item()

                # accuracy
                # (winner takes all)
                _, targets_hat = torch.max(targets_hat.data, 1)
                running_accuracy += (targets_hat == targets).sum().item()
                n_predictions += targets_hat.size(0)

        return running_loss / len(valid_loader), 100 * (running_accuracy / n_predictions)

    def train(self, n_epochs, adj_mat,
              model, optimizer, criterion,
              train_dataloader, valid_dataloader,
              file_path_save_trained_model, file_path_save_best_acc_model, results_file_path,
              train_loss_name, valid_loss_name, accuracy_name,
              best_accuracy_is_max=True):
        """
            Training main entry point.
        """
        print("Starting training...\n")

        # sending to device
        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
        print("The model will be running on", next(
            model.parameters()).device, "device.\n")

        results = []
        best_accuracy = 0.0

        for epoch in range(1, n_epochs + 1):

            epoch_accuracy = 0.0
            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0

            train_epoch_loss = self.train_step(
                model, adj_mat, train_dataloader, optimizer, criterion)
            valid_epoch_loss, epoch_accuracy = self.valid_step(
                model, adj_mat, valid_dataloader, criterion)

            print(f'Epoch: {epoch}/{n_epochs}, {train_loss_name}: {train_epoch_loss:.4f}, {valid_loss_name}: {valid_epoch_loss:.4f}, {accuracy_name}: {epoch_accuracy:.2f}%')

            # saving reached best model
            if best_accuracy_is_max:
                if epoch_accuracy > best_accuracy:
                    #save_checkpoint(model, optimizer, epoch, file_path_save_best_acc_model)
                    best_accuracy = epoch_accuracy
            else:
                if epoch_accuracy < best_accuracy:
                    #save_checkpoint(model, optimizer, epoch, file_path_save_best_acc_model)
                    best_accuracy = epoch_accuracy

            results.append(
                (train_epoch_loss, valid_epoch_loss, epoch_accuracy))

        # Saving the model
        #print('Saving the model...\n')
        #model = model.to('cpu')
        #save_model(model, file_path_save_trained_model)

        # Saving the performances
        # with open(results_file_path, 'wb') as f:
        #    pkl.dump(results, f)

        print("Training finish.\n")

        return model, optimizer
