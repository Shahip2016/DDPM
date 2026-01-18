import torch
from modules import UNet
from diffusion import Diffusion
import logging

logging.basicConfig(level=logging.INFO)

def test_components():
    logging.info("Testing Components...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1. Test UNet
    logging.info("Testing UNet initialization...")
    try:
        model = UNet(device=device).to(device)
        x = torch.randn(2, 3, 64, 64).to(device)
        t = torch.randint(0, 1000, (2,)).to(device)
        out = model(x, t)
        assert out.shape == x.shape, f"UNet output shape mismatch: {out.shape} vs {x.shape}"
        logging.info("UNet Passed ✅")
    except Exception as e:
        logging.error(f"UNet Failed: {e}")

    # 2. Test Diffusion Forward Process
    logging.info("Testing Diffusion Forward Process...")
    try:
        diffusion = Diffusion(img_size=64, device=device)
        x = torch.randn(2, 3, 64, 64).to(device)
        t = diffusion.sample_timesteps(2).to(device)
        x_t, noise = diffusion.noise_images(x, t)
        assert x_t.shape == x.shape, f"Noised image shape mismatch: {x_t.shape} vs {x.shape}"
        assert noise.shape == x.shape, f"Noise shape mismatch: {noise.shape} vs {x.shape}"
        logging.info("Diffusion Forward Process Passed ✅")
    except Exception as e:
        logging.error(f"Diffusion Forward Process Failed: {e}")

    # 3. Test Diffusion Sampling (Reverse Process) - Dry Run
    logging.info("Testing Diffusion Sampling (Dry Run)...")
    try:
        # Reduced noise steps for quick test
        diffusion_quick = Diffusion(noise_steps=10, img_size=32, device=device) 
        model_quick = UNet(c_in=3, c_out=3, time_dim=256, device=device).to(device)
        samples = diffusion_quick.sample(model_quick, n=1)
        assert samples.shape == (1, 3, 32, 32), f"Sampled image shape mismatch: {samples.shape}"
        logging.info("Diffusion Sampling Passed ✅")
    except Exception as e:
        logging.error(f"Diffusion Sampling Failed: {e}")

if __name__ == "__main__":
    test_components()
