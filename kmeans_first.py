#これは2次元データをもとに疑似コードを再現したkmeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs

# 距離関数
def dist(c, s, metric='euclidean'):
    if metric == 'euclidean':
        return np.linalg.norm(c - s)
    elif metric == 'manhattan':
        return np.sum(np.abs(c - s))
    elif metric == 'cosine':
        return cosine_distances([c], [s])[0][0]
    else:
        raise ValueError(f"未知の距離関数です: {metric}")

# セントロイド更新
def update_centroid(c, s, n):
    return c + (1 / n) * (s - c)

# k-means++ 初期化
def initialize_centroids(X_data, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_data)
    return kmeans.cluster_centers_

# クラスタリングアルゴリズム（履歴記録つき）
def clustering_algorithm(X_stream, k, is_correct_fn, metric='euclidean', initial_X_data_for_kmeans_init=None):
    if initial_X_data_for_kmeans_init is None:
        raise ValueError("initial_X_data_for_kmeans_init must be provided for KMeans++ initialization.")

    C = initialize_centroids(initial_X_data_for_kmeans_init, k)
    N = np.zeros(k)
    centroid_history = [C.copy()]

    for S in X_stream:
        dists = [dist(c, S, metric) for c in C]
        min_c = np.argmin(dists)
        N[min_c] += 1

        if is_correct_fn(S, min_c):
            C[min_c] = update_centroid(C[min_c], S, N[min_c])
        centroid_history.append(C.copy())

    return C, N, centroid_history

# 正解判定（教師あり）
def is_correct_fn_factory(true_centers):
    def is_correct(S, min_c):
        dists = [np.linalg.norm(c - S) for c in true_centers]
        correct_cluster = np.argmin(dists)
        return min_c == correct_cluster
    return is_correct

def main(k_clusters: int = 3):
    # データ生成
    X, y_true = make_blobs(n_samples=300, centers=k_clusters, cluster_std=1.0, random_state=42)
    true_centers = [np.mean(X[y_true == i], axis=0) for i in range(k_clusters)]

    # クラスタリング実行（セントロイド履歴付き）
    C_final, N, history = clustering_algorithm(
        X_stream=iter(X),
        k=k_clusters, # k の値を k_clusters に変更
        is_correct_fn=is_correct_fn_factory(true_centers),
        metric='euclidean',
        initial_X_data_for_kmeans_init=X
    )

    # アニメーション用可視化
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.3)
    centers_scatter = ax.scatter([], [], c='red', s=100, marker='X')

    def update(frame):
        current_frame = min(frame, len(history) - 1)
        centers = history[current_frame]
        centers_scatter.set_offsets(centers)
        ax.set_title(f"ステップ {current_frame + 1} / {len(history)} (k={k_clusters})") # タイトルにもkを表示
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True, repeat=False)
    plt.show()

if __name__ == '__main__':
    # ここで k の値を指定して main 関数を呼び出す
    # 例: k=4 で実行
    main(k_clusters=4)

    # 例: k=5 で実行したい場合はコメントアウトを外して実行
    # main(k_clusters=5)
