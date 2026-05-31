# Generative Models in Computer Vision

Discriminative models learn $P(y|x)$ — given an image, predict a label. **Generative models** learn $P(x)$ — model the distribution of images itself, enabling sampling of new images, density estimation, and representation learning.

---

## Why Generative Models?

```
Applications:
  Image synthesis:    "generate a realistic face"
  Data augmentation:  generate extra training examples
  Image editing:      "remove the car from this photo"
  Representation:     learn features without labels
  Density estimation: "how likely is this image under the model?"
  Compression:        encode images efficiently
```

Three families: **VAEs** (encoder-decoder, latent space), **GANs** (adversarial training), **autoregressive models** (explicit density). Diffusion models are covered in [diffusion_models_intro.md](diffusion_models_intro.md).

---

## Variational Autoencoders (VAEs)

### The core idea

Map images to a **latent space** $z$ (low-dimensional, structured), then decode back. The latent space should be continuous and smooth — nearby points decode to similar images, and any point in latent space decodes to something valid.

```
Encoder q_φ:  x → z   (image → latent code, stochastic)
Decoder p_θ:  z → x̂  (latent code → reconstructed image)

Regular autoencoder: encode to a point z = μ
VAE:                 encode to a distribution z ~ N(μ, σ²)
```

### The ELBO

We want to maximize $\log p_\theta(x)$ (make real images likely under our model). This is intractable directly. Instead, derive a lower bound (the **Evidence Lower BOund**):

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)}\left[\log p_\theta(x|z)\right]}_{\text{reconstruction term}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization term}}$$

- **Reconstruction term:** the decoder should reconstruct $x$ well given $z$ sampled from the encoder
- **KL term:** the encoder's distribution over $z$ should be close to the prior $p(z) = \mathcal{N}(0,I)$

This is the ELBO — maximize it as a surrogate for the true log-likelihood.

### Reparameterization Trick

We need to backpropagate through sampling $z \sim \mathcal{N}(\mu, \sigma^2)$. Sampling is non-differentiable. Solution: reparameterize:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now gradients flow through $\mu$ and $\sigma$ (which the encoder outputs), while $\epsilon$ is just a fixed noise source.

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400, latent=20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.fc_mu    = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder  = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim), nn.Sigmoid())
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps          # differentiable!
    
    def forward(self, x):
        mu, logvar = self.encode(x.flatten(1))
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
    def loss(self, x, x_hat, mu, logvar, beta=1.0):
        recon = F.binary_cross_entropy(x_hat, x.flatten(1), reduction='sum')
        kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return recon + beta * kl
```

### VAE Latent Space

After training, the latent space is organized and **interpolatable**:

```
Sample z₁ → decode → image A (smiling woman)
Sample z₂ → decode → image B (frowning man)
Interpolate: decode(α·z₁ + (1-α)·z₂) → smooth transition A→B
```

**β-VAE:** increase the KL weight ($\beta > 1$) to force a more disentangled latent space — individual dimensions correspond to independent factors of variation (smile, hair color, pose).

**VAE limitations:** generated images are often blurry (the model averages over possible reconstructions). GANs produce sharper images.

---

## Generative Adversarial Networks (GANs)

### The Game

Train two networks adversarially:
- **Generator** $G_\theta$: maps random noise $z \sim p(z)$ to fake images
- **Discriminator** $D_\phi$: classifies images as real or fake

```
Real images → D → P(real) → high  ← D wants this
Fake: G(z)  → D → P(real) → low   ← D wants this, G wants opposite

Generator's goal:    make D(G(z)) → 1 (fool the discriminator)
Discriminator's goal: D(real) → 1, D(G(z)) → 0
```

### Minimax Objective

$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

### Training Loop

```python
for real_batch in dataloader:
    z = torch.randn(B, latent_dim)
    fake = G(z)
    
    # --- Train Discriminator ---
    d_real = D(real_batch)
    d_fake = D(fake.detach())    # detach: don't backprop through G
    loss_D = -torch.log(d_real).mean() - torch.log(1 - d_fake).mean()
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()
    
    # --- Train Generator ---
    d_fake = D(fake)             # re-evaluate (D was just updated)
    loss_G = -torch.log(d_fake).mean()  # maximize D(G(z))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

### Architecture Diagram: DCGAN

```
Generator (noise → image):
  z (100,) → FC → reshape (512, 4, 4)
  → ConvTranspose 4×4 (256, 8, 8)   + BN + ReLU
  → ConvTranspose 4×4 (128, 16, 16) + BN + ReLU
  → ConvTranspose 4×4 (64, 32, 32)  + BN + ReLU
  → ConvTranspose 4×4 (3, 64, 64)   + Tanh
  → 64×64 RGB image

Discriminator (image → real/fake):
  3×64×64 → Conv 4×4 (64, 32, 32)   + LeakyReLU
          → Conv 4×4 (128, 16, 16)  + BN + LeakyReLU
          → Conv 4×4 (256, 8, 8)   + BN + LeakyReLU
          → Conv 4×4 (512, 4, 4)   + BN + LeakyReLU
          → FC → sigmoid → real/fake score
```

### GAN Training Problems

**Mode collapse:** the generator finds a few images that fool the discriminator and generates only those — diversity collapses.

```
Data distribution:              Generator after mode collapse:
   ●●● ●●●●● ●●●●               ●●●●●●●●●●●●●●●●●●●●●
   (dogs, cats, cars)            (generates only cats)
```

**Training instability:** the minimax game may not converge. D and G can oscillate without making progress.

**Vanishing gradients:** if D is too good, $D(G(z)) \approx 0$ everywhere → $\log(1 - D(G(z))) \approx 0$ → no gradient for G.

### WGAN: Wasserstein GAN

**The root cause of GAN instability:** JS divergence (implicit in the original GAN) is undefined when distributions don't overlap — which is common early in training.

**Solution:** use **Wasserstein distance** (Earth Mover's distance) instead. It's well-defined even for non-overlapping distributions:

$$W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

The discriminator now outputs a score (not a probability), and its weights are clipped to enforce the Lipschitz constraint (or use gradient penalty in WGAN-GP).

**WGAN advantages:** more stable training, meaningful loss metric (lower loss = better generator), no mode collapse.

---

## Autoregressive Models

### PixelCNN (2016)

**Idea:** model $P(x) = \prod_i P(x_i | x_{<i})$ explicitly. Generate pixel by pixel, left-to-right, top-to-bottom.

```
P(x) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ...

For RGB: P(R,G,B at pixel i | all previous pixels + R,G at pixel i)
```

**Masked convolutions:** at each pixel, the convolution can only see pixels that came before it (masked to zero).

```
Mask A (for first channel):     Mask B (for subsequent channels):
■ ■ ■                           ■ ■ ■
■ ■ □ ← center pixel           ■ ■ ■ ← center pixel
□ □ □                           □ □ □

■ = connection allowed
□ = masked (zero)
```

Advantages: exact likelihood estimation, no mode collapse. Disadvantages: slow generation (one pixel at a time for 256×256 = 65536 steps).

---

## Evaluation Metrics

**Fréchet Inception Distance (FID):** compare distribution of real vs. generated images in feature space:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$

where $\mu, \Sigma$ are mean and covariance of InceptionNet features. Lower = better. FID = 0 means identical distributions.

**Inception Score (IS):** measures sharpness (high confidence per image) and diversity (variety across images). Higher = better.

```
FID scores (ImageNet 256×256, lower is better):
BigGAN:           7.4
StyleGAN2:        3.8
Diffusion (ADM):  2.1   ← diffusion now dominates generation quality
```
