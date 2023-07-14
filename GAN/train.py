import torch
import argparse
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class GANTrainer:
    def __init__(self, args):
        # Set the device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store the command-line arguments
        self.args = args

        # Initialize the generator, discriminator, and dataloader
        self.generator = None
        self.discriminator = None
        self.dataloader = None

    def preprocess_data(self):
        """
        Preprocess the data: convert to tensors and normalize.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # Load the MNIST dataset with the defined transformations
        dataset = datasets.MNIST(root=self.args.dataset_root, train=True, transform=transform, download=True)

        # Create a dataloader for batch processing
        self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

    def initialize_models(self):
        """
        Initialize the generator and discriminator models.
        """
        # Define the shape of the input image
        image_shape = (self.args.image_channle, self.args.image_height, self.args.image_width)

        # Create an instance of the Generator model
        self.generator = Generator(self.args.input_dim, image_shape).to(self.device)

        # Create an instance of the Discriminator model
        self.discriminator = Discriminator(image_shape).to(self.device)

    def train(self):
        """
        Train the GAN models.
        """
        # Define the binary cross-entropy loss function
        criterion = torch.nn.BCELoss()

        # Create Adam optimizers for the generator and discriminator models
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr)

        # Set the models to training mode
        self.generator.train()
        self.discriminator.train()

        # Iterate over the epochs
        for epoch in range(self.args.epochs):
            # Iterate over the batches in the dataloader
            for index, (real_data, _) in enumerate(self.dataloader):
                # Move the data and labels to the device (GPU or CPU)
                real_data = real_data.to(self.device)
                real_label = torch.ones(real_data.size(0), 1).to(self.device)
                fake_label = torch.zeros(real_data.size(0), 1).to(self.device)

                # Reset the gradients of the generator optimizer
                optimizer_G.zero_grad()

                # Generate random noise
                z = torch.randn(real_data.size(0), self.args.input_dim).to(self.device)

                # Generate fake data using the generator model
                fake_data = self.generator(z)

                # Calculate the generator loss and perform backpropagation
                loss_g = criterion(self.discriminator(fake_data), real_label)
                loss_g.backward()
                optimizer_G.step()

                # Reset the gradients of the discriminator optimizer
                optimizer_D.zero_grad()

                # Calculate the discriminator loss using real and fake data
                loss_d_real = criterion(self.discriminator(real_data), real_label)
                loss_d_fake = criterion(self.discriminator(fake_data.detach()), fake_label)

                # Calculate the overall discriminator loss as the average of real and fake losses
                loss_d = (loss_d_fake + loss_d_real) / 2
                loss_d.backward()
                optimizer_D.step()

            # Print the current epoch's loss
            print('Epoch [{}/{}], Loss_G: {:.4f}, Loss_D: {:.4f}'.format(epoch + 1, self.args.epochs, loss_g.item(),
                                                                         loss_d.item()))

    def run(self):
        """
        Run the GAN training process.
        """
        # Preprocess the data
        self.preprocess_data()

        # Initialize the generator and discriminator models
        self.initialize_models()

        # Start the training loop
        self.train()


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    # Create an instance of the ArgumentParser
    parser = argparse.ArgumentParser()

    # Add the desired command-line arguments
    parser.add_argument("--dataset_root", type=str, default='./data', help='dataset path')
    parser.add_argument("--batch_size", type=int, default=64, help='size of batches')
    parser.add_argument("--epochs", type=int, default=100, help='number of model training')
    parser.add_argument("--lr", type=float, default=0.0002, help='learning rate')
    parser.add_argument("--image_channle", type=int, default=1, help='number of image channels')
    parser.add_argument("--image_width", type=int, default=28, help='width size of image')
    parser.add_argument("--image_height", type=int, default=28, help='height size of image')
    parser.add_argument("--input_dim", type=int, default=100, help='dimensionality of the input random noise')

    # Parse the arguments and return the object
    return parser.parse_args()


# Parse the command-line arguments
args = parse_arguments()

# Create an instance of the GANTrainer class with the parsed arguments
gan_trainer = GANTrainer(args)

# Run the GAN training process
gan_trainer.run()
