
# Simple_Instruct_Pix2Pix
Simple Instruct pix2pix is a simple neural network built upon instruct-pix2pix .with the objective
of fine-tunning and inference for image-to-image model.
## Pre-requisite 
Installing requirements.txt
```bash
pip install -r requirements.txt
```
## Command to run fine-tuning
```bash
python fine_tuning_instruct.py
```

## Set training config 
1. set hf_token (hugging face token)
## Dataset Details
1. Product Sketch images (containing edges)
2. Real product images
3. Text Instructions
## Some thoughts on pix2pix.
1. Pix2Pix is a Generative Adversarial Network, or GAN, model 
2. The GAN architecture is comprised of a generator model for outputting new plausible synthetic images, and a discriminator model that classifies images as real (from the dataset) or fake (generated)
3. The discriminator is a deep convolutional neural network that performs image classification. Specifically, conditional-image classification. It takes both the source image (e.g. sketch product photo) and the target image (e.g. real product image) as input and predicts the likelihood of whether target image is real or a fake translation of the source image.PatchGAN model is used.
4. The generator is an encoder-decoder model using a U-Net architecture. The model takes a source image  and generates a target image . It does this by first downsampling or encoding the input image down to a bottleneck layer, then upsampling or decoding the bottleneck representation to the size of the output image. The U-Net architecture means that skip-connections are added between the encoding layers and the corresponding decoding layers, forming a U-shape.
5. The discriminator model is trained directly on real and generated images.
6. The generator model is trained via the discriminator model. It is updated to minimize the loss predicted by the discriminator for generated images marked as “real.” As such, it is encouraged to generate more real images. 

## Components 
1. Variational Auto-Encoder (VAE):
The VAE is used for encoding and decoding images to and from latent representations. It’s a crucial part of the model that allows for the manipulation of images in a controlled latent space.For example in this repo. AutoencoderKL used as VAE.
2. CLIP Text Encoder:
The text encoder from CLIP (Contrastive Language-Image Pretraining) is used for encoding the textual instructions. CLIP is designed to understand text and images in a similar embedding space, which helps the model to interpret the editing instructions accurately.CLIPTextModel is used in the code.
3. Conditional U-Net:
A U-Net architecture is used for denoising the encoded image latents. It’s a type of convolutional network that is particularly effective for tasks like image segmentation and, in this case, image editing.UNet2DConditionModel is used a conditional U-Net.
4. Diffusion Process:
The diffusion process is a generative model that starts with a distribution of noise and gradually denoises it to produce an image. InstructPix2Pix uses this process, conditioned on both the input image and the textual instructions, to generate the edited image.
5. Clip Tokenizer and DDPMScheduler (noise scheduler) is also used in code.

### Code Explanation: Instruct Pix2Pix
##### motivation to go ahead :
You copied that function without understanding why it does what it does, and as a result your code IS GARBAGE. AGAIN. from Linus Torvalds

##### 1. Importing Libraries
The code imports various libraries needed for working with image editing using diffusion models and transformers:
- **`diffusers`**: Contains classes for working with diffusion models.
- **`transformers`**: Provides tools for working with text models like CLIP.

##### 2. Setting Up Components
- **Noise Scheduler**: Helps manage the noise added during the training process.
- **Tokenizer**: Converts text into a format the model can understand.
- **Text Encoder**: Encodes the input text into embeddings (numerical representations).
- **VAE (Variational Autoencoder)**: Encodes and decodes images into a latent space (a compressed version of the image).
- **UNet**: A neural network architecture used for image generation tasks.

##### 3. Model
- **Model Details**: Consists of vae,text encoder,unet components inside tranformers StableDiffusionInstructPix2PixPipeline class.
gpu offloading sequence text encoder->unet->vae
- **Freeze VAE and Text Encoder**: These parts of the model won't be trained further.
- **U-Net**: Trainable = True

##### 4. Creating an Exponential Moving Average (EMA) Model
An EMA model helps stabilize training by averaging the weights of the UNet over time.

##### 5. Optimizer Setup
The optimizer is set up to adjust the weights of the UNet during training.

##### 6. Loading the Dataset
The dataset containing images and text prompts is loaded. The dataset has three columns:
- **Original Images**: The images before editing.
- **Edit Prompts**: Text descriptions of the edits to be made.
- **Edited Images**: The images after editing.

##### 7. Preprocessing Functions
These functions process the images and text to prepare them for training:
- **`preprocess_images`**: Converts images to a format suitable for the model and applies transformations.
- **`preprocess_train`**: Prepares both original and edited images along with their text prompts.
- **`collate_fn`**: Gathers batches of data to feed into the model during training.
                    Function return original_pixel_values , edited_pixel_values , input_ids
            
##### 8. Creating DataLoader
The DataLoader helps manage the dataset and batches during training.


##### 10. Training Loop
This is where the actual training happens:
- For each epoch (a full pass through the dataset):
  - The model is set to training mode.
  - For each batch of data:
    - Images are encoded to a latent space.
    - Noise is added to these latents.
    - Text prompts are encoded.
    - The original image embeddings are also encoded.
    - Conditional dropout is applied to simulate real-world variations.
    - Noisy latents and original image embeddings are concatenated.
    - The model predicts the noise added earlier.
    - Loss (the difference between predicted and actual noise) is calculated.
    - The model's weights are adjusted based on the loss.

##### 10.1 Detail Explanation For Fine-Tuning

##### 1. Setting Up for Training
- **`unet.train()`**: This sets the UNet model to training mode. This is important because some layers behave differently during training and evaluation (like dropout layers).
- **`train_loss = 0.0`**: Initialize a variable to keep track of the total training loss for the current epoch.

##### 2. Encoding Images to Latent Space
We want to learn the denoising process w.r.t the edited images which are conditioned on the original image (which was edited) and the edit instruction.
- **`latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype))`**: Encodes the edited images into latent representations.
- **`latent_dist.sample()`**: Samples from the latent distribution.
- **`latents = latents * vae.config.scaling_factor`**: Scales the latent representations.


##### 3. Adding Noise to Latents
- **`noise = torch.randn_like(latents)`**: Generates random noise with the same shape as the latents.
- **`timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)`**: Samples random timesteps for each image.
- **`noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)`**: Adds noise to the latents according to the timestep.

##### 4. Encoding Text Prompts
- **`encoder_hidden_states = text_encoder(batch["input_ids"])[0]`**: Encodes the text prompts into embeddings.

##### 5. Encoding Original Images
- **`original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()`**: Encodes the original images into latent representations.


##### 6. Applying Conditional Dropout
-Conditional Dropout for Text:
- **`random_p = torch.rand(bsz, device=latents.device, generator=generator)`**: Generates random probabilities.
- **`prompt_mask = random_p < 2 * args.conditioning_dropout_prob`**: Creates a mask for the text prompts.
- **`null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]`**: Gets the embedding for an empty prompt.
- **`encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)`**: Applies the conditional dropout.

-Conditional Dropout for Images:
- **`image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * args.- conditioning_dropout_prob).to(image_mask_dtype))`**: Creates a mask for the original images.
- **`original_image_embeds = image_mask * original_image_embeds`**: Applies the conditional dropout.


##### 7. Concatenating Noisy Latents and Original Image Embeddings
- **`concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)`**: Concatenates the noisy latents with the original image embeddings.

##### 8. Calculating the Target for Loss
- Determining Target Based on Prediction Type:
- **`target = noise`**

##### 9. Predicting Noise and Calculating Loss
- **`model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample`**: The model predicts the noise.encoder_hidden_states = text_encoder(batch["input_ids"])[0]

- **`loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")`**: Calculates the mean squared error loss between the predicted and actual noise.


##### 10. Avg. Loss and Backpropagation
-Averaging Loss Across Processes:
- **`avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()`**: Averages the loss across all processes (if using distributed training).
- **`train_loss += avg_loss.item() / args.gradient_accumulation_steps`**: Accumulates the training loss.

-Backpropagation:
- **`accelerator.backward(loss)`**: Performs backpropagation to compute gradients.
- **`accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)`**: Clips gradients to prevent exploding gradients.
- **`optimizer.step()`**: Updates model weights.
- **`lr_scheduler.step()`**: Updates learning rate.
- **`optimizer.zero_grad()`**: Resets gradients for the next iteration.
