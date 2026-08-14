# pytorch_training

Personal ML learning repo. Working through vision model fundamentals from scratch using PyTorch, starting at the beginning and building up.

## What this is

Not a polished project. This is a learning progression. Each folder is a stage in understanding how computers learn to classify images, starting with the simplest possible problem and adding complexity as the concepts click.

The goal was never just to get the accuracy number up. It was to understand *why* each architecture decision exists and what problem it solves that the previous one couldn't.

## Progression

### mnist_classifier
First model. Fully connected network on the MNIST handwritten digit dataset. Gets the job done on simple grayscale digits but has no spatial awareness. It treats every pixel independently, which works here but falls apart fast when images get more complex.

### mnist_cnn
Same dataset, convolutional architecture. The jump from fully connected to CNN was the first real "oh, that's why" moment. Instead of looking at every pixel in isolation, the conv layers scan for patterns across regions of the image. Spatial relationships start to matter.

### cifar10
Where it gets interesting. MNIST is grayscale digits: clean, simple, forgiving. CIFAR-10 is 10 classes of real color images: planes, cars, birds, cats, and so on. Three color channels instead of one, and a lot more variation to handle.

The thing that surprised me about conv layers here is that they work by essentially distorting the image in a predictable way. Each filter mangles the input differently, and those transformations surface abstract patterns like edges, textures, and shapes that the network can actually learn from. The image that comes out the other side of a conv layer looks nothing like what went in, but it's more useful to the model than the original was. Each layer transforms in its own way and hands something different downstream.

Got to the point of diminishing returns on conv layer depth. Currently exploring learning rate scheduling and data augmentation as the next levers.

## What I've learned so far

- How a train/test loop actually works and why the split matters
- The difference between loss as memorization vs genuine generalization, and why chasing accuracy on training data is a trap
- Why convolutional layers exist and what they're actually doing to the image
- How to guard against overfitting through architecture choices
- Where the gains stop coming from adding more conv layers and what to try next

## Stack

Python, PyTorch, torchvision. Virtual environments not included. Install your own with `pip install torch torchvision`.

Data downloads automatically on first run via torchvision datasets.
