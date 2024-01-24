import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#-----------------------------------------------------------------------

def adversarial_example_class(image, epsilon):
    
    gradient = image.grad.data

    # Create adversarial image
    adversarial_example = image + (epsilon * gradient.sign())
    adversarial_output = model(adversarial_example.unsqueeze(0))
    adversarial_prediction = torch.argmax(adversarial_output).item()

    return adversarial_prediction

#-------------------------------------------------------------------------

def gen_adversarial_example(image, model):

    # Keep gradients
    image.requires_grad = True

    # Original image class
    output = model(image.unsqueeze(0))
    original_prediction = torch.argmax(output).item()

    # Calculate the loss
    loss = F.cross_entropy(output, torch.tensor([original_prediction]))
    model.zero_grad()

    # Backward pass to compute the gradient of the loss with respect to the input image
    loss.backward()
    gradient = image.grad.data

    # Increment epsilon until classification is wrong
    epsilon = 0
    while adversarial_example_class(image, epsilon) == original_prediction:
        epsilon += 0.01

    adversarial_example = image + (epsilon * gradient.sign())

    return np.round(epsilon, 2), adversarial_example

#-------------------------------------------------------------------------

def plot_adv_example(image, model):

    # Original Prediction
    output = model(image.unsqueeze(0))
    original_prediction = torch.argmax(output).item()

    # Get the confidence associated with the original prediction
    image_probabilities = F.softmax(output, dim=1)
    confidence = image_probabilities[0, original_prediction].item()

    # Adversarial prediction
    eps, adversarial_image = gen_adversarial_example(image, model)

    # Verify the model's prediction on the adversarial example
    adversarial_output = model(adversarial_image.unsqueeze(0))
    adversarial_prediction = torch.argmax(adversarial_output).item()

    # Get the confidence associated with the adversarial prediction
    adv_image_probabilities = F.softmax(adversarial_output, dim=1)
    adv_confidence = adv_image_probabilities[0, adversarial_prediction].item()

    # Convert tensor to numpy arrays
    original_image_np = image.squeeze().detach().numpy()
    adversarial_image_np = adversarial_image.squeeze().detach().numpy()

    # Display the original and adversarial images side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=1, right=0.5)

    # Original Image
    axes[0].imshow(original_image_np, cmap='gray')
    axes[0].set_title('Original')
    axes[0].set_xlabel(f'{labels_dict[original_prediction]}\n{confidence*100:.1f}%')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Noise
    noise = adversarial_image - image
    grad = noise/eps
    noise_output = model(grad.unsqueeze(0))
    noise_prediction = torch.argmax(noise_output).item()
    noise_np = noise.squeeze().detach().numpy()

    # Get the confidence associated with the noise prediction
    noise_probabilities = F.softmax(noise_output, dim=1)
    noise_confidence = noise_probabilities[0, noise_prediction].item()

    axes[1].imshow(noise_np, cmap='gray')
    axes[1].set_title('Noise')
    axes[1].set_xlabel(f'{labels_dict[noise_prediction]}\n{noise_confidence*100:.1f}%')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Adversarial Example
    axes[2].imshow(adversarial_image_np, cmap='gray')
    axes[2].set_title('Adversarial')
    axes[2].set_xlabel(f'{labels_dict[adversarial_prediction]}\n{adv_confidence*100:.1f}%')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Add a simple plus sign between the images
    axes[0].text(1.58, 0.5, '+   ' + str(eps) + ' x', ha="center", va="center", fontsize=10, transform=axes[0].transAxes)
    #axes[0].text(1.8, 0.5, '(' + str(eps) + ') x ', ha="center", va="center", fontsize=2, transform=axes[0].transAxes)
    axes[1].text(1.5, 0.5, '=', ha="center", va="center", fontsize=20, transform=axes[1].transAxes)

    plt.show()