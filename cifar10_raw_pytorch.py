import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from absl import app
from absl import flags
from wrn_model import WideResNet
from sngp import SNGPClassifier, SNGP
import torch.optim as optim
from sngp_helpers import evaluate, compute_accuracy_ece


FLAGS = flags.FLAGS


def get_output_shape(model, input_shape):
    x = torch.randn(1, *input_shape)
    out = model(x)
    shape = tuple(out.shape[1:])
    assert len(shape) == 1     
    return shape[0]


def load_data():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    datapath = "/cephyr/users/walline/Alvis/NOBACKUP/torchvision_data"

    trainset = torchvision.datasets.CIFAR10(root=datapath, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch, shuffle=True)
    
    
    testset = torchvision.datasets.CIFAR10(root=datapath, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch, shuffle=False)
    
    return trainloader, testloader

def eval_accuracy(predictions, labels):

    max_predictions = predictions.argmax(axis=1)
    accuracy = (max_predictions == labels).sum()/labels.size
    return accuracy

def main(*args):

    del args
    
    trainloader, testloader = load_data()

    cifar_shape = (3, 32, 32)
    n_classes = 10
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    WRN = WideResNet
    hidden_out_shape = get_output_shape(WRN(depth=FLAGS.depth, widen_factor=FLAGS.widen_factor), cifar_shape)

    use_spectral_norm = True if FLAGS.gp else False
    
    model = SNGP(hidden_map=WRN,
                 gp_input_size=hidden_out_shape,
                 gp_hidden_size=1024,
                 gp_output_size=n_classes,
                 gp_scale=FLAGS.gp_scale,
                 gp_s=FLAGS.gp_s,
                 gp_m=FLAGS.gp_m,
                 use_gp_layer=FLAGS.gp,
                 use_spectral_norm=use_spectral_norm,
                 input_size=cifar_shape,
                 depth=FLAGS.depth,
                 widen_factor=FLAGS.widen_factor)

    criterion = nn.CrossEntropyLoss()

    params_config = [{"params": model.hidden_map.parameters(), "weight_decay": FLAGS.wd_hidden},
                     {"params": model.output_layer.parameters(), "weight_decay": FLAGS.wd_output_layer}]
        
    optimizer = optim.SGD(params_config, lr=FLAGS.lr, momentum=0.9)
    
    # training loop
    model.to(device)

    for epoch in range(FLAGS.epochs):

        #print("RUNNING EPOCH {} OF {}".format(epoch+1, FLAGS.epochs))
        model.train()
        for i, data in enumerate(trainloader):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        # eval mean prediction accuracy
        mean_predictions = []
        all_labels = []

        model.eval()        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)

                mean_predictions.append(model.mean_predictions(images))
                all_labels.append(labels)

            mean_predictions = torch.cat(mean_predictions, axis=0).cpu().numpy()
            all_labels = torch.cat(all_labels, axis=0).cpu().numpy()

        acc = eval_accuracy(mean_predictions, all_labels)
        print("Epoch nr: {} - mean accuracy: {}".format(epoch+1, acc), flush=True)
                
    # on train end
    model.eval()
    model.compute_full_precisions(trainloader, device)    
    model.compute_covariances()
    print("Finished training")

    # test loop

    predictions = []
    
    with torch.no_grad():
        for data in testloader:
            
            images = data[0].to(device)
            predictions.append(model.predict_proba(images))


        predictions = torch.cat(predictions, axis=0).cpu().numpy()

        
    results = evaluate(all_labels, predictions)
    [print("{} : {}".format(key, value)) for key, value in results.items()]

    accuracy, ece = compute_accuracy_ece(predictions, all_labels, 10)
    print("other function: accuracy: {} - ece: {}".format(accuracy, ece))
    
        
if __name__ == "__main__":

    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("wd_hidden", 0.0005, "Weight decay for hidden mapping")
    flags.DEFINE_float("wd_output_layer", 0.0, "Weight decay for final layer")
    flags.DEFINE_float("gp_scale", 2.0, "GP length scale parameter")
    flags.DEFINE_float("gp_s", 0.001, "s: covariance ridge factor")
    flags.DEFINE_float("gp_m", 0.99, "m: covariance discount factor")
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_integer("epochs", 1, "Number of training epochs")
    flags.DEFINE_boolean("gp", True, "Use GP as output layer")
    flags.DEFINE_integer("depth", 28, "WRN depth")
    flags.DEFINE_integer("widen_factor", 2, "WRN widen factor")

    app.run(main)
    
