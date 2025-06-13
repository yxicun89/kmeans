import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from sklearn.datasets import make_blobs, make_moons, make_circles

# --- 特徴量の重みを定義 ---
FEATURE_WEIGHTS = np.array([
    1.0, # connected_components
    1.0, # loop_statements
    1.0, # conditional_statements
    1.0, # cycles
    1.0, # paths
    1.0, # cyclomatic_complexity
    0.6, # variable_count
    0.1, # total_reads
    0.1, # total_writes
    0.1, # max_reads
    0.1  # max_writes
])

# --- 距離関数（重み付きユークリッド距離） ---
def dist(c, s, metric='euclidean', weights=None):
    if metric == 'euclidean':
        if weights is None:
            return np.linalg.norm(c - s)
        else:
            return np.sqrt(np.sum(weights * (c - s)**2))
    elif metric == 'manhattan':
        if weights is None:
            return np.sum(np.abs(c - s))
        else:
            return np.sum(weights * np.abs(c - s))
    elif metric == 'cosine':
        if weights is None:
            return cosine_distances([c], [s])[0][0]
        else:
            c_w = c * np.sqrt(weights)
            s_w = s * np.sqrt(weights)
            return cosine_distances([c_w], [s_w])[0][0]
    else:
        raise ValueError(f"未知の距離関数です: {metric}")

# --- セントロイド更新 ---
def update_centroid(c, s, n):
    return c + (1 / n) * (s - c)

# --- k-means++ 初期化 ---
def initialize_centroids(X_data, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_data)
    return kmeans.cluster_centers_

# --- クラスタリングアルゴリズム（履歴記録つき） - 標準的なK-means版 ---
def clustering_algorithm_standard_kmeans(X_stream, k, metric='euclidean', initial_X_data_for_kmeans_init=None, weights=None):
    C = initialize_centroids(initial_X_data_for_kmeans_init, k)
    N = np.zeros(k)
    centroid_history = [C.copy()]

    for S in X_stream:
        dists = [dist(c, S, metric, weights=weights) for c in C]
        min_c = np.argmin(dists)

        N[min_c] += 1
        C[min_c] = update_centroid(C[min_c], S, N[min_c])

        centroid_history.append(C.copy())

    return C, N, centroid_history

# --- データ生成関数（is_correct_fn_factory は今回は使用しないが、残しておく） ---
def is_correct_fn_factory(true_centers):
    def is_correct(S, min_c):
        if len(true_centers) == 0:
             return True
        dists = [np.linalg.norm(c - S) for c in true_centers]
        if not dists:
            return True
        correct_cluster_idx = np.argmin(dists)
        if min_c < len(true_centers):
            return min_c == correct_cluster_idx
        else:
            return False
    return is_correct

# --- 各種データ生成関数（変更なし） ---
def generate_sparse_clusters(n_per_cluster=40, noise_ratio=0.2, seed=42):
    np.random.seed(seed)
    base_centers = np.array([
        [1, 0, 1, 0, 1, 2, 2, 1, 1, 1, 1],
        [3, 2, 4, 3, 5, 8, 4, 2, 2, 2, 2],
        [1, 1, 2, 1, 2, 3, 10, 20, 20, 10, 10]
    ])
    data = []
    labels = []
    for i, center in enumerate(base_centers):
        samples = np.random.normal(loc=center, scale=1.5, size=(n_per_cluster, len(center)))
        samples = np.round(samples).astype(int)
        samples = np.clip(samples, 0, None)
        data.append(samples)
        labels.extend([i] * n_per_cluster)
    
    n_noise = int(len(labels) * noise_ratio)
    noise = np.random.randint(0, 25, size=(n_noise, base_centers.shape[1]))
    data.append(noise)
    labels.extend([-1] * n_noise)

    X = np.vstack(data)
    y = np.array(labels)
    
    return X, y, base_centers, "Sparse Clusters with Noise"

def generate_moon_data(n_samples=300, noise=0.05, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y, [], "Moon-shaped Data"

def generate_circle_data(n_samples=300, noise=0.04, factor=0.5, seed=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
    return X, y, [], "Circle-shaped Data"

def generate_uneven_blobs(n_samples=500, seed=42):
    centers = [[-5, -5], [0, 0], [5, 5]]
    cluster_std = [0.8, 2.5, 0.5]
    n_per_cluster = [n_samples // 4, n_samples // 2, n_samples // 4]

    X_data = []
    y_data = []
    for i, center in enumerate(centers):
        X_blob = np.random.normal(loc=center, scale=cluster_std[i], size=(n_per_cluster[i], 2))
        X_data.append(X_blob)
        y_data.extend([i] * n_per_cluster[i])

    X = np.vstack(X_data)
    y = np.array(y_data)
    true_centers = np.array(centers)
    return X, y, true_centers, "Uneven Blobs"

def generate_linear_data(n_samples=300, seed=42):
    np.random.seed(seed)
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    X[:n_samples//2, 0] = np.linspace(0, 10, n_samples//2) + np.random.normal(0, 0.5, n_samples//2)
    X[:n_samples//2, 1] = np.linspace(0, 10, n_samples//2) + np.random.normal(0, 0.5, n_samples//2)
    y[:n_samples//2] = 0

    X[n_samples//2:, 0] = np.linspace(0, 10, n_samples - n_samples//2) + np.random.normal(0, 0.5, n_samples - n_samples//2)
    X[n_samples//2:, 1] = np.linspace(10, 0, n_samples - n_samples//2) + np.random.normal(0, 0.5, n_samples - n_samples//2)
    y[n_samples//2:] = 1

    true_centers = np.array([X[y==0].mean(axis=0), X[y==1].mean(axis=0)])
    return X, y, true_centers, "Linear Data"

def generate_ideal_blobs(n_samples=300, n_clusters=3, cluster_std=1.0, seed=42):
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std, random_state=seed)
    true_centers = np.array([X[y==i].mean(axis=0) for i in range(n_clusters)])
    return X, y, true_centers, "Ideal Blobs"

# --- メイン関数 ---
def main(k_clusters: int = 3, draw_convex_hull: bool = True, draw_decision_boundary: bool = False, 
         data_generator_fn = generate_sparse_clusters, generator_args: dict = None):
    
    if generator_args is None:
        generator_args = {}

    X, y_true, true_centers_for_eval, data_description = data_generator_fn(**generator_args)
    
    n_features = X.shape[1] 

    if n_features == 2:
        X_2d = X
        true_centers_2d = np.array(true_centers_for_eval) if len(true_centers_for_eval) > 0 else np.array([])
        is_pca_applied = False
    else:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        true_centers_2d = pca.transform(true_centers_for_eval) if len(true_centers_for_eval) > 0 else np.array([])
        is_pca_applied = True
    
    C_final, N, history = clustering_algorithm_standard_kmeans(
        X_stream=iter(X),
        k=k_clusters,
        metric='euclidean',
        initial_X_data_for_kmeans_init=X,
        weights=FEATURE_WEIGHTS if n_features == 11 else None
    )

    history_2d = []
    if is_pca_applied:
        history_2d = [pca.transform(h) for h in history]
    else:
        history_2d = history

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.get_cmap('viridis', k_clusters) 
    
    if draw_decision_boundary:
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]

        C_final_2d_for_boundary = history_2d[-1]

        labels_predict = np.array([np.argmin([np.linalg.norm(p - c_2d) for c_2d in C_final_2d_for_boundary]) for p in mesh_points_2d])
        labels_predict = labels_predict.reshape(xx.shape)

        ax.contourf(xx, yy, labels_predict, cmap=cmap, alpha=0.1)

    if draw_convex_hull:
        unique_labels_for_hull = [l for l in np.unique(y_true) if l != -1]
        
        for label in unique_labels_for_hull:
            points_in_cluster = X_2d[y_true == label]
            if len(points_in_cluster) >= 3:
                try:
                    hull = ConvexHull(points_in_cluster)
                    for simplex in hull.simplices:
                        ax.plot(points_in_cluster[simplex, 0], points_in_cluster[simplex, 1], 
                                 '--', color=cmap(label % k_clusters), alpha=0.7)
                except Exception as e:
                    print(f"Warning: Could not draw Convex Hull for true cluster {label}: {e}")
            else:
                print(f"Warning: Not enough points ({len(points_in_cluster)}) to draw Convex Hull for true cluster {label}.")

    unique_labels_data = np.unique(y_true)
    for label in unique_labels_data:
        if label == -1:
            ax.scatter(X_2d[y_true == label, 0], X_2d[y_true == label, 1], 
                       c='gray', alpha=0.3, label='Noise')
        else:
            ax.scatter(X_2d[y_true == label, 0], X_2d[y_true == label, 1], 
                       color=cmap(label % k_clusters), label=f"True Cluster {label}", alpha=0.5)
    
    if len(true_centers_2d) > 0:
        ax.scatter(true_centers_2d[:, 0], true_centers_2d[:, 1], 
                   c='blue', s=200, marker='o', edgecolors='black', label='True Centers (Base)')

    centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label=f'Calculated Centroids (k={k_clusters})')
    
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Standard Online K-means (PCA 2D Projection) - K-means k={k_clusters} on {data_description}")
    plt.legend()
    plt.grid(True)

    def update(frame):
        current_frame = min(frame, len(history_2d) - 1)
        centers_2d = history_2d[current_frame]
        centers_scatter.set_offsets(centers_2d)
        ax.set_title(f"ステップ {current_frame + 1} / {len(history_2d)} (K-means k={k_clusters}) on {data_description}")
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
    plt.show()

    print(f"最終的なセントロイド（K-meansのk={k_clusters}）:\n", np.round(C_final, 2))
    print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")
    print(f"データセット: {data_description}")
    print(f"K-meansは{k_clusters}個のクラスタを見つけようとしました。")


if __name__ == '__main__':
    # 単一のデータセットでの実行
    main(k_clusters=4, draw_convex_hull=True, draw_decision_boundary=True, 
         data_generator_fn=generate_sparse_clusters)