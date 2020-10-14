import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class SpatialMNIST(Dataset):
  def __init__(self, data_dir, train):
    super(SpatialMNIST, self).__init__()

    self.data = datasets.MNIST(
        data_dir,
        train=train,
        download=True,
        transform=transforms.ToTensor())
    self.grid = np.stack(np.meshgrid(range(28),
                                     range(27, -1, -1)),
                         axis=-1).reshape([-1, 2])

  def __getitem__(self, idx):
    img, label = self.data[idx]
    img = img.numpy().reshape([-1])
    p = img / img.sum()
    replace = True if (sum(p > 0) < 64) else False
    ind = np.random.choice(784, 64, p=p, replace=replace)
    x = self.grid[ind].copy().astype(np.float32)
    # x += np.random.uniform(0., 1., (64, 2))

    return x, label

  def __len__(self):
    return len(self.data)


class TVSpatialMNIST(SpatialMNIST):
  def __init__(
      self, data_dir, train,
      num_times=51, max_time=1., fixed_time=True,
      p_action=0., p_rotate=0., p_translate=0., p_magnify=0.):
    super(TVSpatialMNIST, self).__init__(data_dir=data_dir, train=train)

    self.max_time = max_time
    self.fixed_time = fixed_time

    self.num_times = num_times
    self.p_rotate = p_rotate
    self.p_translate = p_translate
    self.p_magnify = p_magnify
    self.p_action = p_action

  def get_time(self):
    if self.fixed_time:
      max_time = self.max_time
    else:
      raise NotImplementedError()

    return np.linspace(0., max_time, self.num_times, dtype=np.float32)

  def __getitem__(self, idx):
    x, label = super(TVSpatialMNIST, self).__getitem__(idx)
    x = x.astype(np.float32)
    x -= np.mean(x, axis=0, keepdims=True)
    x = np.repeat(np.expand_dims(x, axis=0), repeats=self.num_times, axis=0)

    t = self.get_time()
    x, omega = self.rotate(t, x)

    x += np.random.uniform(0., 0.2, (self.num_times, 64, 2))
    x /= 14.0

    return t, x, label  # TxKxD

  def rotate(self, t, x):
    omega = np.pi
    theta = t * omega

    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array(((cos, -sin), (sin, cos))).transpose(2, 0, 1)

    return x @ R, omega

  # def magnify(self, t, x):
  #   mag = np.exp(np.random.normal(scale=0.25)) * t
  #   return x * np.expand_dims(np.expand_dims(mag, axis=1), axis=1), mag

  # def translate(self, t, x):
  #   v = np.random.normal(size=(1, 1, 2), scale=2.).astype(np.float32)
  #   return x + v * np.expand_dims(np.expand_dims(t, axis=1), axis=1), v


def get_loader(is_time_varying, data_dir, batch_size=128, num_workers=2,
               num_times=51, max_time=1,
               p_action=1., p_rotate=1., p_magnify=0, p_translate=0):
  if is_time_varying:
    trainset = TVSpatialMNIST(
        data_dir=data_dir, train=True,
        num_times=num_times, max_time=max_time, fixed_time=True,
        p_action=p_action, p_rotate=p_rotate, p_translate=p_translate,
        p_magnify=p_magnify)
    testset = TVSpatialMNIST(
        data_dir=data_dir, train=False,
        num_times=num_times, max_time=max_time, fixed_time=True,
        p_action=p_action, p_rotate=p_rotate, p_translate=p_translate,
        p_magnify=p_magnify)
  else:
    trainset = SpatialMNIST(data_dir=data_dir, train=True)
    testset = SpatialMNIST(data_dir=data_dir, train=False)


  trainloader = DataLoader(trainset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers)
  testloader = DataLoader(testset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers)

  return trainloader, testloader


if __name__ == '__main__':
    train_loader, test_loader = get_loader(is_time_varying=True, data_dir='./data', batch_size=10, num_times=5)
    t, x = next(iter(train_loader))
    print(x.shape)
    print(x)