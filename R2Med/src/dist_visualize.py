import matplotlib.pyplot as plt
import pickle as pkl

if __name__ == '__main__':
    # load data
    with open('./hnrr_bm25_medical_k100.pkl', 'rb') as f:
        hnrr_bm25 = pkl.load(f)
    with open('./true_hnrr_bm25_medical_k100.pkl', 'rb') as f:
        true_hnrr_bm25 = pkl.load(f)
    with open('./hnrr_bge_medical_k100.pkl', 'rb') as f:
        hnrr_bge = pkl.load(f)
    with open('./true_hnrr_bge_medical_k100.pkl', 'rb') as f:
        true_hnrr_bge = pkl.load(f)

    # figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(hnrr_bm25, bins=20, edgecolor='black')
    axes[0, 0].set_title("General Hard Negative RR - BM25", fontsize=15)
    axes[0, 0].set_xlabel("Rank", fontsize = 15)
    axes[0, 0].set_ylabel("Count", fontsize = 15)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)

    axes[0, 1].hist(true_hnrr_bm25, bins=20, edgecolor='black')
    axes[0, 1].set_title("True Hard Negative RR - BM25", fontsize=15)
    axes[0, 1].set_xlabel("Rank", fontsize = 15)
    axes[0, 1].set_ylabel("Count", fontsize = 15)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=10)

    axes[1, 0].hist(hnrr_bge, bins=20, edgecolor='black')
    axes[1, 0].set_title("General Hard Negative RR - BGE", fontsize=15)
    axes[1, 0].set_xlabel("Rank", fontsize = 15)
    axes[1, 0].set_ylabel("Count", fontsize = 15)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=10)

    axes[1, 1].hist(true_hnrr_bge, bins=20, edgecolor='black')
    axes[1, 1].set_title("True Hard Negative RR - BGE", fontsize=15)
    axes[1, 1].set_xlabel("Rank", fontsize = 15)
    axes[1, 1].set_ylabel("Count", fontsize = 15)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig("hnrr_medical_2x2.pdf")
    plt.close()