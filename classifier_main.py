import torch
import torch.nn as nn
import torch.optim as optim
import gcommand_loader
from Model import Model
import numpy as np


NUM_WORKERS = 20
BATCH_SIZE = 100
LEARNING_RATE = 0.05
EPOCHS = 10

label_names = {0: 'bed', 1: "bird", 2: "cat", 3: "dog", 4: "down", 5: "eight", 6: "five",
               7: "four", 8: "go", 9: "happy", 10: "house", 11: "left", 12: "marvin", 13: "nine",
               14: "no", 15: "off", 16: "on", 17: "one", 18: "right", 19: "seven",
               20: "sheila", 21: "six", 22: "stop", 23: "three", 24: "tree", 25: "two",
               26: "up", 27: "wow", 28: "yes", 29: "zero"}


def train_model(model_optimizer_lossfunc_device, train_set):
    model = model_optimizer_lossfunc_device[0]
    optimizer = model_optimizer_lossfunc_device[1]
    loss_function = model_optimizer_lossfunc_device[2]
    device = model_optimizer_lossfunc_device[3]

    model.train()  # training mode
    for index, (example, label, path) in enumerate(train_set):
        example, label = example.to(device), label.to(device)
        optimizer.zero_grad()
        prediction = model(example)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()


def test_model(model_optimizer_lossfunc_device, test_set):
    model = model_optimizer_lossfunc_device[0]
    loss_function = model_optimizer_lossfunc_device[2]
    device = model_optimizer_lossfunc_device[3]

    model.eval()  # evaluation mode
    loss = 0
    accuracy = 0
    for test_example, target, path in test_set:
        test_example, target = test_example.to(device), target.to(device)
        model_output = model(test_example)
        loss += loss_function(model_output, target).item()
        prediction = model_output.data.max(1, keepdim=True)[1]
        accuracy += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    loss /= len(test_set.dataset)
    accuracy = 100. * accuracy / len(test_set.dataset)


def train_and_validate_model(model_optimizer_lossfunc_device, train_set, validation_set):
    for epoch in range(0, EPOCHS):
        train_model(model_optimizer_lossfunc_device, train_set)
        test_model(model_optimizer_lossfunc_device, validation_set)


def get_file_name_from_path(file_path):
    file_name = ""
    for char in reversed(file_path):
        if char == '/':
            break
        file_name += char
    return file_name[::-1]


def get_predictions(model_optimizer_lossfunc_device, test_set):
    model = model_optimizer_lossfunc_device[0]
    device = model_optimizer_lossfunc_device[3]

    model.eval()  # evaluation mode
    predictions = []
    for data, target, path in test_set:
        data, target = data.to(device), target.to(device)
        model_output = model(data)

        prediction = model_output.data.max(1, keepdim=True)[1]
        predicted_label = int(prediction[0][0])

        prediction_by_format = get_file_name_from_path(path[0]) + ", " + label_names[predicted_label]
        predictions.append(prediction_by_format)
    return np.array(predictions)


def write_labels_to_file(labels):
    with open("test_y", "w") as labels_file:
        for label in labels:
            labels_file.write(label)
            labels_file.write('\n')


def main():
    # load the train set from files
    train_set = gcommand_loader.GCommandLoader("train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS, pin_memory=True, sampler=None)
    # load the validation set from files
    validation_set = gcommand_loader.GCommandLoader("valid")
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=None,
                                                    num_workers=NUM_WORKERS, pin_memory=True, sampler=None)
    # load the test set from files
    test_set = gcommand_loader.GCommandLoader("test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=None,
                                              num_workers=NUM_WORKERS, pin_memory=True, sampler=None)

    # check if GPU is available and if it is - switch the device to use it
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # create a new model, optimizer and loss func, and switch to using available device
    model = Model()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    model_optimizer_lossfunc_device = (model, optimizer, loss_function, device)
    # start training and validation process
    train_and_validate_model(model_optimizer_lossfunc_device, train_loader, validation_loader)

    # get the predictions on the test set and write them to file
    labels = get_predictions(model_optimizer_lossfunc_device, test_loader)
    write_labels_to_file(labels)


if __name__ == "__main__":
    main()
