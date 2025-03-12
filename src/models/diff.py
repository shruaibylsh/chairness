import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from accelerate import Accelerator
from torch.optim import AdamW
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chair Generation with Diffusion Model Transfer Learning")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing chair images")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory to save results")
    parser.add_argument("--image_size", type=int, default=128, 
                        help="Image size for training and generation")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=15, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--save_every", type=int, default=5, 
                        help="Save model every N epochs")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="Number of data loader workers")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and use existing model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to existing model checkpoint for generation or continued training")
    return parser.parse_args()

class ChairDataset(Dataset):
    """Dataset for chair images."""
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Find all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_files.extend(
                [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                 if f.lower().endswith(ext)]
            )
        
        # Define transformations
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        
        return image

def setup_diffusion_model(image_size=128):
    """Set up the pre-trained diffusion model for transfer learning."""
    # Initialize the U-Net model from a pre-trained checkpoint
    model = UNet2DModel.from_pretrained(
        "google/ddpm-celebahq-256",
        use_auth_token=False
    )
    
    # Update the model configuration to match our image size
    model.config.sample_size = image_size
    
    # Freeze the first few layers to preserve learned features
    params_to_freeze = []
    for name, param in model.named_parameters():
        if "down_blocks.0" in name or "down_blocks.1" in name:
            param.requires_grad = False
            params_to_freeze.append(name)
    
    print(f"Frozen parameters: {len(params_to_freeze)}/{len(list(model.named_parameters()))}")
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    
    return model, noise_scheduler

def show_samples(dataset, num_samples=5, save_path=None):
    """Display sample images from the dataset."""
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        # Get a random image
        idx = random.randint(0, len(dataset)-1)
        img = dataset[idx]
        
        # Denormalize
        img = img * 0.5 + 0.5
        
        # Convert to numpy and transpose for plotting
        img = img.numpy().transpose(1, 2, 0)
        
        # Plot
        axs[i].imshow(img)
        axs[i].axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Sample images saved to {save_path}")
    plt.close()

def generate_sample_images(model, noise_scheduler, num_images, image_size, output_path, device):
    """Generate sample images during training to monitor progress."""
    with torch.no_grad():
        # Create pipeline for inference
        pipeline = DDPMPipeline(
            unet=model,
            scheduler=noise_scheduler
        )
        
        # Move pipeline to device
        pipeline = pipeline.to(device)
        
        # Generate images
        images = pipeline(
            batch_size=num_images,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images
        
        # Plot and save images
        fig, axs = plt.subplots(2, num_images//2, figsize=(15, 10))
        axs = axs.flatten()
        
        for i, image in enumerate(images):
            # Convert to numpy 
            image_np = np.array(image)
            
            # Plot
            axs[i].imshow(image_np)
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Generated sample images saved to {output_path}")

def train_diffusion_model(
    model,
    noise_scheduler,
    dataset,
    num_epochs=30,
    learning_rate=1e-5,
    batch_size=16,
    gradient_accumulation_steps=1,
    save_model_epochs=5,
    image_size=128,
    results_dir="results",
    device=None,
    num_workers=2
):
    """Fine-tune the diffusion model on our chair dataset."""
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    
    # Prepare for training with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    global_step = 0
    
    # Create folder for saved images and models
    os.makedirs(f"{results_dir}/diffusion_chairs/samples", exist_ok=True)
    os.makedirs(f"{results_dir}/diffusion_chairs/models", exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (bs,), device=clean_images.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Get model prediction
            with accelerator.accumulate(model):
                # The model predicts the noise to be removed
                noise_pred = model(
                    noisy_images, 
                    timesteps
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backpropagate the gradients
                accelerator.backward(loss)
                
                # Clip gradients for stability
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update the model parameters
                optimizer.step()
                optimizer.zero_grad()
            
            # Update the progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            
            # Generate and save sample images every 500 steps
            if global_step % 500 == 0:
                generate_sample_images(
                    accelerator.unwrap_model(model),
                    noise_scheduler,
                    4,
                    image_size,
                    f"{results_dir}/diffusion_chairs/samples/step_{global_step}.png",
                    device
                )
        
        # Generate and save sample images each epoch
        generate_sample_images(
            accelerator.unwrap_model(model),
            noise_scheduler,
            8,
            image_size,
            f"{results_dir}/diffusion_chairs/samples/epoch_{epoch+1}.png",
            device
        )
        
        # Save model checkpoint
        if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
            # Unwrap model
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Save model
            torch.save(
                unwrapped_model.state_dict(),
                f"{results_dir}/diffusion_chairs/models/model_epoch_{epoch+1}.pt"
            )
            
            # Create pipeline for inference and save it
            pipeline = DDPMPipeline(
                unet=unwrapped_model,
                scheduler=noise_scheduler
            )
            pipeline.save_pretrained(f"{results_dir}/diffusion_chairs/models/pipeline_epoch_{epoch+1}")
    
    return accelerator.unwrap_model(model), noise_scheduler

def generate_final_images(model, noise_scheduler, num_images=16, image_size=128, results_dir="results", device=None):
    """Generate a grid of final chair images using our trained model."""
    # Create output directory
    output_dir = f"{results_dir}/diffusion_chairs/final_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = DDPMPipeline(
        unet=model,
        scheduler=noise_scheduler
    )
    pipeline = pipeline.to(device)
    
    # Use different seeds for variety
    all_images = []
    for i in range(num_images // 4):
        # Generate images with different seeds
        seed = i * 1000 + 42
        images = pipeline(
            batch_size=4,
            generator=torch.Generator(device=device).manual_seed(seed)
        ).images
        all_images.extend(images)
    
    # Create a grid display
    rows = num_images // 4
    cols = 4
    
    plt.figure(figsize=(15, rows * 4))
    for i, image in enumerate(all_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.array(image))
        plt.axis('off')
        
        # Save individual images
        image.save(f"{output_dir}/chair_{i+1}.png")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_grid.png")
    plt.close()
    
    print(f"Generated {num_images} chair images")
    print(f"Images saved to {output_dir}")
    
    return all_images

def generate_clean_images(model, noise_scheduler, num_images=8, image_size=128, results_dir="results", device=None):
    """Generate exceptionally clean chair images with minimal noise."""
    # Create output directory
    output_dir = f"{results_dir}/diffusion_chairs/clean_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline with more inference steps for cleaner results
    pipeline = DDPMPipeline(
        unet=model,
        scheduler=noise_scheduler
    )
    pipeline = pipeline.to(device)
    
    # Override the scheduler with more diffusion steps for cleaner results
    pipeline.scheduler.config.num_train_timesteps = 1000
    
    # Use different seeds for variety but with consistent settings for clean results
    all_images = []
    for i in range(num_images):
        # Generate image with a specific seed
        seed = i * 1000 + 100
        with torch.no_grad():
            # Generate with higher number of inference steps for cleaner results
            image = pipeline(
                batch_size=1,
                generator=torch.Generator(device=device).manual_seed(seed),
                num_inference_steps=200  # More steps = smoother results
            ).images[0]
            
            all_images.append(image)
            
            # Save individual image
            image.save(f"{output_dir}/clean_chair_{i+1}.png")
    
    # Create a grid display
    rows = 2
    cols = num_images // 2
    
    plt.figure(figsize=(15, 8))
    for i, image in enumerate(all_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.array(image))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/clean_grid.png")
    plt.close()
    
    print(f"Generated {num_images} clean chair images with minimal noise")
    print(f"Clean images saved to {output_dir}")
    
    return all_images

def main():
    """Main function to run the chair generation pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create the chair dataset
    print(f"Loading dataset from {args.data_dir}...")
    chair_dataset = ChairDataset(args.data_dir, image_size=args.image_size)
    print(f"Dataset contains {len(chair_dataset)} images")
    
    # Show some sample images from the dataset
    show_samples(chair_dataset, num_samples=5, save_path=f"{args.results_dir}/dataset_samples.png")
    
    if args.skip_training and args.checkpoint is None:
        print("Error: --skip_training requires --checkpoint")
        return
    
    if args.checkpoint:
        # Load existing model and scheduler
        print(f"Loading model from checkpoint: {args.checkpoint}")
        pipeline = DDPMPipeline.from_pretrained(args.checkpoint)
        model = pipeline.unet
        noise_scheduler = pipeline.scheduler
        model.to(device)
    else:
        # Initialize the diffusion model
        print("Initializing diffusion model...")
        model, noise_scheduler = setup_diffusion_model(args.image_size)
        model.to(device)
    
    if not args.skip_training:
        # Train the model
        print(f"Training for {args.num_epochs} epochs...")
        trained_model, trained_scheduler = train_diffusion_model(
            model,
            noise_scheduler,
            chair_dataset,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            save_model_epochs=args.save_every,
            image_size=args.image_size,
            results_dir=args.results_dir,
            device=device,
            num_workers=args.num_workers
        )
        print("Training complete!")
    else:
        trained_model = model
        trained_scheduler = noise_scheduler
    
    # Generate final chair images
    print("Generating final images...")
    generate_final_images(
        trained_model, 
        trained_scheduler, 
        num_images=16, 
        image_size=args.image_size,
        results_dir=args.results_dir,
        device=device
    )
    
    # Generate low-noise chair images
    print("Generating low-noise images...")
    generate_clean_images(
        trained_model, 
        trained_scheduler, 
        num_images=8, 
        image_size=args.image_size,
        results_dir=args.results_dir,
        device=device
    )

if __name__ == "__main__":
    main()