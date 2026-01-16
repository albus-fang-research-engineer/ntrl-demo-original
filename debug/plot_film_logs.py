import torch
import matplotlib.pyplot as plt
import numpy as np

def main(log_path):
    film_log = torch.load(log_path)

    epoch = np.array(film_log["epoch"])
    batch = np.array(film_log["batch"])

    gamma_mean = np.array(film_log["gamma_mean"])
    gamma_std  = np.array(film_log["gamma_std"])
    beta_mean  = np.array(film_log["beta_mean"])
    beta_std   = np.array(film_log["beta_std"])

    # Convert (epoch, batch) → global step index
    steps = np.arange(len(gamma_mean))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # --- Gamma ---
    axs[0, 0].plot(steps, gamma_mean)
    axs[0, 0].axhline(1.0, linestyle="--", color="gray")
    axs[0, 0].set_title("FiLM γ mean")
    axs[0, 0].set_ylabel("γ")

    axs[1, 0].plot(steps, gamma_std)
    axs[1, 0].set_title("FiLM γ std")
    axs[1, 0].set_ylabel("σ(γ)")
    axs[1, 0].set_xlabel("Training step")

    # --- Beta ---
    axs[0, 1].plot(steps, beta_mean)
    axs[0, 1].axhline(0.0, linestyle="--", color="gray")
    axs[0, 1].set_title("FiLM β mean")
    axs[0, 1].set_ylabel("β")

    axs[1, 1].plot(steps, beta_std)
    axs[1, 1].set_title("FiLM β std")
    axs[1, 1].set_ylabel("σ(β)")
    axs[1, 1].set_xlabel("Training step")

    plt.tight_layout()
    plt.savefig("FiLM.png")
    plt.show()


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2, "Usage: python plot_film_logs.py path/to/film_log.pt"
    main(sys.argv[1])
