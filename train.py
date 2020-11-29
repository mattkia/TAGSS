from __future__ import division
from __future__ import print_function

import time
import torch
import datetime
import torch.optim as optim
from data_loader import BuildDataSet, DataSetV2, DataSetV3
from models import IPoolGC, TopologyAwareGSSGC
from sklearn.model_selection import KFold
from operator import itemgetter


# ------------------------------ Training settings ------------------------------

dataset_name = 'FRANKENSTEIN'
n_epochs = 30
loss_pooling_config = False
regularization_factor = 1
optimizer_lr = 0.001

n_hid = 64
n_feat = 780
n_fc1 = 32
n_fc2 = 16
n_class = 2
adj_bar_hop_order = 1
variation_hop_order = 1
pooling_ratio = 0.1

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

batch_size = 60

now = datetime.datetime.now()

config = {
    'loss_pooling': loss_pooling_config,
    'regularization_factor': regularization_factor,
    'optimizer_lr': optimizer_lr,
    'n_hid': n_hid,
    'adj_bar_hop_order': adj_bar_hop_order,
    'variation_hop_order': variation_hop_order,
    'pooling_ratio': pooling_ratio,
    'batch_size': batch_size
}

# ---------------------------------------------------------------------------------

# Model and optimizer

criteria = torch.nn.CrossEntropyLoss()


def train(adjacency, features, label):
    model.train()
    output = model(features, adjacency)
    output = output.view(1, output.size()[0])
    label = label.view(1)
    label = label.type(torch.long)
    loss_train = criteria(output, label)

    return loss_train, output


def evaluate(adjacency, features, label):
    model.eval()
    output = model(features, adjacency)
    output = output.view(1, output.size()[0])
    label = label.view(1)
    label = label.type(torch.long)
    loss_val = criteria(output, label)
    return loss_val.item(), output


def save_checkpoint(state, fold_number):
    if loss_pooling_config:
        filename = f'runs/checkpoints/{dataset_name}/batched_checkpoint_kf{fold_number}_with_loss.pth.tar'
    else:
        filename = f'runs/checkpoints/{dataset_name}/batched_checkpoint_kf{fold_number}.pth.tar'

    return fold_number


data_set_builder = BuildDataSet(dataset_name)
data_set = data_set_builder.get_dataset()

kf = KFold(n_splits=10)

kf_counter = 1

for train_index, test_index in kf.split(data_set):
    model = TopologyAwareGSSGC(n_feat=n_feat, n_hid=n_hid, n_fc1=n_fc1, n_fc2=n_fc2, n_class=n_class,
                               pooling_ratio=pooling_ratio, adj_bar_hop_order=adj_bar_hop_order,
                               variation_hop_order=variation_hop_order, loss=loss_pooling_config)

    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr, weight_decay=0.00001)

    print(f'==================================== FOLD {kf_counter} ==================================================')

    train_idx = train_index[: - int(0.1 * len(train_index))]
    validation_idx = train_index[- int(0.1 * len(train_index)):]
    test_idx = test_index

    data_set_train = list(itemgetter(*train_idx)(data_set))
    data_set_validation = list(itemgetter(*validation_idx)(data_set))
    data_set_test = list(itemgetter(*test_idx)(data_set))

    train_data = DataSetV3(data_set_train)
    validation_data = DataSetV3(data_set_validation)
    test_data = DataSetV3(data_set_test)

    n_batch_train = int(len(train_data) / batch_size)

    # Train model
    total_time = time.time()
    best_val_acc = 0.
    best_kf = kf_counter
    test_acc = 0
    for epoch in range(n_epochs):
        t_total = time.time()
        total_train_loss = 0

        for batch_index in range(n_batch_train):
            batch_train_loss = torch.tensor(0.)
            optimizer.zero_grad()

            for sample in train_data[batch_index * batch_size: (batch_index + 1) * batch_size]:
                adj = sample['adjacency_matrix']
                x = sample['feature_matrix']
                target = sample['label']
                loss, out_train = train(adj, x, target)
                batch_train_loss += loss

            mean_batch_train_loss = batch_train_loss / batch_size
            mean_batch_train_loss.backward()
            optimizer.step()
            total_train_loss += mean_batch_train_loss.item()

        # Train Accuracy
        train_total = 0
        train_count = 0

        for j in range(len(train_data)):
            adj_train = train_data[j]['adjacency_matrix']
            x_train = train_data[j]['feature_matrix']
            target_train = train_data[j]['label']
            _, out_train = evaluate(adj_train, x_train, target_train)

            if torch.argmax(out_train).item() == target_train.item():
                train_count += 1.0

            train_total += 1

        train_acc = (train_count / train_total) * 100

        # Validation Accuracy
        val_total = 0
        val_count = 0
        total_validation_loss = 0

        for j in range(len(validation_data)):
            adj_val = validation_data[j]['adjacency_matrix']
            x_val = validation_data[j]['feature_matrix']
            target_val = validation_data[j]['label']
            loss_validation, out_val = evaluate(adj_val, x_val, target_val)
            total_validation_loss += loss_validation

            if torch.argmax(out_val).item() == target_val.item():
                val_count += 1.0

            val_total += 1

        validation_acc = (val_count / val_total) * 100

        # Remember the Best Accuracy and Save the Checkpoint
        is_best = validation_acc >= best_val_acc
        best_val_acc = max(validation_acc, best_val_acc)

        if is_best:
            kf_best = save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_val_acc,
                'optimizer': optimizer.state_dict(),
            }, kf_counter)

            # Test Accuracy
            test_total = 0
            test_count = 0
            total_test = 0

            for j in range(len(test_data)):
                adj_test = test_data[j]['adjacency_matrix']
                x_test = test_data[j]['feature_matrix']
                target_test = test_data[j]['label']
                _, out_test = evaluate(adj_test, x_test, target_test)

                if torch.argmax(out_test).item() == target_test.item():
                    test_count += 1.0

                test_total += 1

            test_acc = (test_count / test_total) * 100

        print('[*] Epoch:', epoch + 1,
              'train_loss:{0:.2f}'.format(total_train_loss),
              'train_acc:{0:.2f}'.format(train_acc),
              'validation_loss:{0:.2f}'.format(total_validation_loss),
              'validation_acc:{0:.2f}'.format(validation_acc),
              'test_acc: {0:.2f}'.format(test_acc),
              'time:{0:.2f}'.format(time.time() - t_total), 's')

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - total_time))
    kf_counter +=1

