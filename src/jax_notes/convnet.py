import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from datasets import load_dataset

import flax
import flax.nnx
import optax


KAIMING = flax.nnx.initializers.kaiming_normal()  # aka He normal
BIAS_ZERO = flax.nnx.initializers.zeros


class ConvSubblock(flax.nnx.Module):
    def __init__(self, rngs, channels_in, channels_out,
                kernel_size=3, pool_stride=2, num_conv_layers=1, dropout_rate=0.2):
        self.conv = flax.nnx.Conv(
            channels_in, channels_out, (kernel_size, kernel_size), rngs=rngs, padding="SAME", 
            kernel_init=KAIMING, bias_init=BIAS_ZERO
        )
        self.bnorm = flax.nnx.BatchNorm(channels_out, rngs=rngs)
        self.dropout = flax.nnx.Dropout(dropout_rate)

    def __call__(self, x, *, rngs):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.dropout(x, rngs=rngs)
        x = flax.nnx.relu(x)
        return x


class ConvLayer(flax.nnx.Module):
    def __init__(self, rngs, channels_in, channels_out,
                kernel_size=3, pool_stride=2, num_conv_layers=1, dropout_rate=0.2):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.pool_stride = pool_stride
        self.num_conv_layers = num_conv_layers

        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(ConvSubblock(rngs, channels_in, channels_out, kernel_size, pool_stride, num_conv_layers, dropout_rate))
            channels_in = channels_out

        self.conv_layers = flax.nnx.Sequential(*conv_layers)

        self.pool = partial(
            flax.nnx.max_pool, 
            window_shape=(pool_stride, pool_stride), 
            strides=(pool_stride, pool_stride),
            padding="SAME"
        )
        
    def __call__(self, x, *, rngs):
        x = self.conv_layers(x, rngs=rngs)
        x = self.pool(x)
        return x


class MLP(flax.nnx.Module):
    def __init__(self, rngs, input_dim, hidden_dim, num_classes, dropout_rate=0.2):
        self.fc1 = flax.nnx.Linear(
            input_dim, hidden_dim, rngs=rngs, kernel_init=KAIMING, bias_init=BIAS_ZERO
        )
        self.fc2 = flax.nnx.Linear(
            hidden_dim, 10, rngs=rngs, kernel_init=KAIMING, bias_init=BIAS_ZERO
        )
        self.bnorm = flax.nnx.BatchNorm(hidden_dim, rngs=rngs)
        self.dropout = flax.nnx.Dropout(dropout_rate)
    
    def __call__(self, x, *, rngs):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bnorm(x)
        x = flax.nnx.relu(x)
        x = self.dropout(x, rngs=rngs)
        x = self.fc2(x)
        return x


class ConvNet(flax.nnx.Module):
    def __init__(
        self, rngs, image_height, image_width, image_channels, num_classes, num_conv_blocks=3, 
        num_conv_layers_per_block=2, hidden_dim=256, channel_multiplier=4, dropout_rate=0.2
    ):
        self.hidden_dim = hidden_dim
        self.channel_multiplier = channel_multiplier
        self.dropout_rate = dropout_rate

        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.num_classes = num_classes

        channel_sizes = [image_channels * (channel_multiplier**i) for i in range(num_conv_blocks)]

        self.conv_layers = flax.nnx.Sequential(*[
            ConvLayer(rngs, c, c*channel_multiplier, num_conv_layers=num_conv_layers_per_block, dropout_rate=dropout_rate)
            for c in channel_sizes
        ])

        final_channels = channel_sizes[-1] * channel_multiplier
        input_height = image_height
        input_width = image_width
        for _ in range(num_conv_blocks):
            input_height = (input_height + 1) // 2
            input_width = (input_width + 1) // 2
        
        mlp_dim = input_height * input_width * final_channels
        print(f"MLP dim: {mlp_dim}")

        self.mlp = MLP(rngs, mlp_dim, hidden_dim, self.num_classes, dropout_rate)
    
    def __call__(self, x, *, rngs):
        x = x.reshape(x.shape[0], self.image_height, self.image_width, self.image_channels)
        x = self.conv_layers(x, rngs=rngs)
        x = self.mlp(x, rngs=rngs)
        return x


import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def _affine_warp_2d(img, angle_deg, tx, ty, scale, shear_x, shear_y):
    """
    img: (H, W), single channel
    """
    H, W = img.shape
    angle = jnp.deg2rad(angle_deg)

    # Center of the image (x is width, y is height)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    # Rotation, shear, scale matrices (2x2)
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)

    R = jnp.array([[cos, -sin],
                   [sin,  cos]])

    Sh = jnp.array([[1.0,    shear_x],
                    [shear_y, 1.0   ]])

    Sc = jnp.array([[scale, 0.0],
                    [0.0,   scale]])

    # Total linear transform A
    A = R @ Sh @ Sc  # (2, 2)

    # Build output coordinate grid (y, x)
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

    # Shift grid to center
    xs_c = xs - cx
    ys_c = ys - cy

    # Apply A manually: [x'; y'] = A @ [x; y]
    a00, a01 = A[0, 0], A[0, 1]
    a10, a11 = A[1, 0], A[1, 1]

    xs_lin = a00 * xs_c + a01 * ys_c
    ys_lin = a10 * xs_c + a11 * ys_c

    # Shift back + apply translation
    xs_t = xs_lin + cx + tx
    ys_t = ys_lin + cy + ty

    # map_coordinates expects coords ordered as (axis0, axis1) = (y, x)
    sample_coords = jnp.stack([ys_t, xs_t], axis=0)  # (2, H, W)

    warped = map_coordinates(
        img,
        sample_coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return warped


def affine_warp_image(
    img,
    angle_deg,
    tx,
    ty,
    scale,
    shear_x,
    shear_y,
):
    """
    img: (H, W)        for grayscale
         (H, W, C)     for NHWC color (e.g. CIFAR-10)
    """
    if img.ndim == 2:
        # single-channel
        return _affine_warp_2d(img, angle_deg, tx, ty, scale, shear_x, shear_y)
    elif img.ndim == 3:
        H, W, C = img.shape

        # vmap the 2D warp over the channel axis
        def warp_one_channel(ch):
            return _affine_warp_2d(ch, angle_deg, tx, ty, scale, shear_x, shear_y)

        # in_axes=2: iterate over last axis (C), out_axes=2: put channels back in last axis
        warped = jax.vmap(warp_one_channel, in_axes=2, out_axes=2)(img)
        return warped
    else:
        raise ValueError(f"Expected img.ndim 2 or 3, got {img.ndim}")


def augment(
    key,
    images,                      # (B, H, W) or (B, H, W, C)
    percent_augmented=0.5,
    max_translation=2.0,         # in pixels
    max_rotation_deg=15.0,
    max_scale_change=0.1,        # +/- 10% zoom
    max_shear=0.1,               # small shear
):
    """
    Apply random affine transforms (rotation, translation, scale, shear)
    to ~percent_augmented of the batch.

    images: (B, H, W)      e.g. MNIST
            (B, H, W, C)   e.g. CIFAR-10 NHWC
    returns: (key_out, images_out)
    """
    if images.ndim == 3:
        B, H, W = images.shape
        has_channels = False
    elif images.ndim == 4:
        B, H, W, C = images.shape
        has_channels = True
    else:
        raise ValueError(f"augment expects images of shape (B,H,W) or (B,H,W,C), got {images.shape}")

    # Split keys for different random draws
    key, key_mask, key_tx, key_ty, key_rot, key_scale, key_shear = jax.random.split(key, 7)

    # Mask: which examples to augment?
    do_aug = jax.random.bernoulli(
        key_mask, p=percent_augmented, shape=(B,)
    )  # (B,)

    # Translations
    tx = jax.random.uniform(
        key_tx, (B,),
        minval=-max_translation,
        maxval= max_translation,
    )
    ty = jax.random.uniform(
        key_ty, (B,),
        minval=-max_translation,
        maxval= max_translation,
    )

    # Rotations
    angles = jax.random.uniform(
        key_rot, (B,),
        minval=-max_rotation_deg,
        maxval= max_rotation_deg,
    )

    # Scales (zoom in/out)
    scales = 1.0 + jax.random.uniform(
        key_scale, (B,),
        minval=-max_scale_change,
        maxval= max_scale_change,
    )

    # Shears
    shear_x = jax.random.uniform(
        key_shear, (B,),
        minval=-max_shear,
        maxval= max_shear,
    )
    shear_y = jax.random.uniform(
        key_shear, (B,),
        minval=-max_shear,
        maxval= max_shear,
    )

    # Vectorize affine_warp_image over the batch
    batch_warp = jax.vmap(affine_warp_image)
    augmented = batch_warp(
        images, angles, tx, ty, scales, shear_x, shear_y
    )

    # Mix original and augmented according to mask
    if has_channels:
        mask = do_aug[:, None, None, None]  # (B, 1, 1, 1)
    else:
        mask = do_aug[:, None, None]        # (B, 1, 1)

    out = jnp.where(mask, augmented, images)

    return key, out



def compute_whitening_params(images, eps=1e-1):
    images = images.reshape(images.shape[0], -1)
    mean = jnp.mean(images, axis=0)
    centered = images - mean
    cov = (centered.T @ centered) / images.shape[0]

    # now prevent small eigenvalues from causing numerical issues when inverting via solve_triangular below
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals_clamped = jnp.maximum(eigvals, eps)
    cov_sqrt_inv = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals_clamped)) @ eigvecs.T
    return mean, cov_sqrt_inv


@jax.jit
def whiten(image, mean, cov_sqrt_inv):
    image_flat = image.reshape(image.shape[0], -1)
    image_flat_centered = image_flat - mean
    return (image_flat_centered @ cov_sqrt_inv).reshape(image.shape)


def prepare_mnist(key):
    mnist = load_dataset("mnist")

    train_ds = mnist["train"]
    test_ds = mnist["test"]

    train_images_jnp = jnp.array(train_ds["image"]).astype(jnp.float32) / 255.0
    print(f"MNIST Train images shape: {train_images_jnp.shape} -- max: {jnp.max(train_images_jnp)}, min: {jnp.min(train_images_jnp)}")

    key, key_subsample = jax.random.split(key)
    image_subsample = jax.random.choice(key_subsample, train_images_jnp, shape=(10_000,), replace=False)
    print(f"MNIST Image subsample shape: {image_subsample.shape} -- max: {jnp.max(image_subsample)}, min: {jnp.min(image_subsample)}")

    train_mean, train_cov_sqrt = compute_whitening_params(image_subsample)
    print(f"MNIST mean shape: {train_mean.shape} -- min: {jnp.min(train_mean)}, max: {jnp.max(train_mean)}")
    print(f"MNIST cov_sqrt shape: {train_cov_sqrt.shape} -- min: {jnp.min(train_cov_sqrt)}, max: {jnp.max(train_cov_sqrt)}")

    test_images_jnp = jnp.array(test_ds["image"]).astype(jnp.float32) / 255.0
    whitened_train = whiten(train_images_jnp, train_mean, train_cov_sqrt)
    whitened_test = whiten(test_images_jnp, train_mean, train_cov_sqrt)

    print("MNIST training data shape:", whitened_train.shape, "test data shape:", whitened_test.shape)
    print(f"MNIST whitened train min: {jnp.min(whitened_train)}, max: {jnp.max(whitened_train)}")
    print(f"MNIST whitened test min: {jnp.min(whitened_test)}, max: {jnp.max(whitened_test)}")

    # let's convert the labels to one-hot encoding
    train_labels = jnp.array(train_ds["label"])
    test_labels = jnp.array(test_ds["label"])

    print(f"MNIST train labels shape: {train_labels.shape}, test labels shape: {test_labels.shape}")

    return key, whitened_train, whitened_test, train_labels, test_labels, 28, 28, 1, 10


def prepare_cifar10(key):
    cifar10 = load_dataset("uoft-cs/cifar10")

    train_ds = cifar10["train"]
    test_ds = cifar10["test"]

    train_images_jnp = jnp.array(train_ds["img"]).astype(jnp.float32) / 255.0
    print(f"CIFAR-10 Train images shape: {train_images_jnp.shape} -- max: {jnp.max(train_images_jnp)}, min: {jnp.min(train_images_jnp)}")
    
    key, key_subsample = jax.random.split(key)
    image_subsample = jax.random.choice(key_subsample, train_images_jnp, shape=(10_000,), replace=False)
    print(f"CIFAR-10 Image subsample shape: {image_subsample.shape} -- max: {jnp.max(image_subsample)}, min: {jnp.min(image_subsample)}")

    train_mean, train_cov_sqrt = compute_whitening_params(image_subsample)
    print(f"CIFAR-10 mean shape: {train_mean.shape} -- min: {jnp.min(train_mean)}, max: {jnp.max(train_mean)}")
    print(f"CIFAR-10 cov_sqrt shape: {train_cov_sqrt.shape} -- min: {jnp.min(train_cov_sqrt)}, max: {jnp.max(train_cov_sqrt)}")
    
    test_images_jnp = jnp.array(test_ds["img"]).astype(jnp.float32) / 255.0
    whitened_train = whiten(train_images_jnp, train_mean, train_cov_sqrt)
    whitened_test = whiten(test_images_jnp, train_mean, train_cov_sqrt)

    print("CIFAR-10 training data shape:", whitened_train.shape, "test data shape:", whitened_test.shape)
    print(f"CIFAR-10 whitened train min: {jnp.min(whitened_train)}, max: {jnp.max(whitened_train)}")
    print(f"CIFAR-10 whitened test min: {jnp.min(whitened_test)}, max: {jnp.max(whitened_test)}")
    
    # let's convert the labels to one-hot encoding
    train_labels = jnp.array(train_ds["label"])
    test_labels = jnp.array(test_ds["label"])

    print(f"CIFAR-10 train labels shape: {train_labels.shape}, test labels shape: {test_labels.shape}")

    return key, whitened_train, whitened_test, train_labels, test_labels, 32, 32, 3, 10



def flax_convent_loss(conv_net, rngs, batch):
    images, labels = batch
    logits = conv_net(images, rngs=rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits

FLAX_PASS_MODEL_TO_OPTIMIZER = tuple([int(x) for x in flax.__version__.split(".")[:2]]) >= (0, 11)

@flax.nnx.jit
def flax_train_step_conv_net(conv_net, optimizer, metrics, rngs, batch, key):
    """Train for a single step."""
    images, labels = batch
    key, images = augment(key, images)
    grad_fn = flax.nnx.value_and_grad(flax_convent_loss, has_aux=True)
    (loss, logits), grads = grad_fn(conv_net, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=labels)  # In-place updates.
    if FLAX_PASS_MODEL_TO_OPTIMIZER:
        optimizer.update(conv_net, grads)  # In-place updates.
    else:
        optimizer.update(grads)  # In-place updates.


@flax.nnx.jit
def flax_convnet_eval_step(conv_net, metrics, rngs, batch):
    images, labels = batch
    loss, logits = flax_convent_loss(conv_net, rngs, batch)
    metrics.update(loss=loss, logits=logits, labels=labels)  # In-place updates.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--train-steps", type=int, default=32000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-conv-blocks", type=int, default=3)
    parser.add_argument("--num-conv-layers-per-block", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--channel-multiplier", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.2)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    key = jax.random.key(args.seed)

    prepare_fn = prepare_mnist
    if args.dataset == "cifar10":
        prepare_fn = prepare_cifar10
    elif args.dataset != "mnist":
        raise ValueError(f"Invalid dataset: {args.dataset}")

    key, train_images, test_images, train_labels, test_labels, image_height, image_width, image_channels, num_classes = prepare_fn(key)

    rngs = flax.nnx.Rngs(42)
    conv_net = ConvNet(
        rngs, 
        image_height,
        image_width,
        image_channels,
        num_classes,
        num_conv_blocks=args.num_conv_blocks, 
        num_conv_layers_per_block=args.num_conv_layers_per_block, 
        hidden_dim=args.hidden_dim, 
        channel_multiplier=args.channel_multiplier, 
        dropout_rate=args.dropout_rate
    )
        
    flax.nnx.display(conv_net)

    learning_rate = args.learning_rate
    momentum = 0.9

    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.99,
    )

    optimizer = flax.nnx.Optimizer(
        conv_net, 
        optax.adamw(schedule, args.momentum), 
        wrt=flax.nnx.Param
    )

    metrics = flax.nnx.MultiMetric(
        accuracy=flax.nnx.metrics.Accuracy(),
        loss=flax.nnx.metrics.Average("loss"),
    )

    flax.nnx.display(optimizer)

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }

    eval_every = args.eval_every
    train_steps = args.train_steps
    batch_size = args.batch_size

    rngs = flax.nnx.Rngs(0)

    for step in range(train_steps):
        key, key_subsample = jax.random.split(key)
        indices = jax.random.randint(key_subsample, (batch_size,), 0, train_images.shape[0])
        batch = (train_images[indices], train_labels[indices])
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics
        conv_net.train() # Switch to train mode
        flax_train_step_conv_net(conv_net, optimizer, metrics, rngs, batch, key)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            conv_net.eval() # Switch to eval mode
            flax_convnet_eval_step(conv_net, metrics, rngs, (test_images, test_labels))

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

            # Plot loss and accuracy in subplots
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            # ax1.set_title('Loss')
            # ax2.set_title('Accuracy')
            # for dataset in ('train', 'test'):
            #    ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            #    ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
            # ax1.legend()
            # ax2.legend()

            train_loss = metrics_history['train_loss'][-1]
            train_acc = metrics_history['train_accuracy'][-1]
            test_loss = metrics_history['test_loss'][-1]
            test_acc = metrics_history['test_accuracy'][-1] 
            lr = schedule(optimizer.step.value)
            print(f"Step {step} LR: {lr:.4e}, train loss: {train_loss:.4f}, acc: {train_acc:.4f}, test loss: {test_loss:.4f}, acc: {test_acc:.4f}")
