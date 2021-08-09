import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from absl import app
from absl import flags
from wrn_model import WideResNet
from sngp import SNGPClassifier, SNGP

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
    train_imgs = []
    train_labels = []
    for img, label in trainset:
        train_imgs.append(img)
        train_labels.append(label)
    train_imgs = torch.stack(train_imgs, dim=0)
    
    testset = torchvision.datasets.CIFAR10(root=datapath, train=False, transform=transform)
    test_imgs = []
    test_labels = []
    for img, label in testset:
        test_imgs.append(img)
        test_labels.append(label)
    test_imgs = torch.stack(test_imgs, dim=0)

    return train_imgs, train_labels, test_imgs, test_labels
    

def main(*args):

    del args
    
    train_imgs, train_labels, test_imgs, test_labels = load_data()

    cifar_shape = tuple(train_imgs.shape[1:])
    
    n_classes = 10
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    depth=10
    widen_factor=1
    
    WRN = WideResNet
    hidden_out_shape = get_output_shape(WRN(depth=depth, widen_factor=widen_factor), cifar_shape)
    
    sngp = SNGPClassifier(
        module=SNGP,
        module__hidden_map=WRN,
        module__gp_input_size=hidden_out_shape,
        module__gp_hidden_size=1024,
        module__gp_output_size=n_classes,
        module__use_spectral_norm=True,
        module__use_gp_layer=FLAGS.gp,
        module__input_size=cifar_shape,
        module__depth=depth,
        module__widen_factor=widen_factor,
        criterion=nn.CrossEntropyLoss,
        optimizer__lr=FLAGS.lr,
        optimizer__weight_decay=FLAGS.wd,
        max_epochs=FLAGS.epochs,
        batch_size=FLAGS.batch,
        train_split=None,
        device=device
    )

    sngp.fit(train_imgs, train_labels)
    
    print(sngp.score(test_imgs, test_labels))
    print(sngp.mean_prediction_score(test_imgs, test_labels))



if __name__ == "__main__":

    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("wd", 0.0005, "Weight decay")
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_integer("epochs", 10, "Number of training epochs")
    flags.DEFINE_boolean("gp", True, "Use GP as output layer")
    
    app.run(main)
    
