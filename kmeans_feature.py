# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_distances
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # 特徴量名（例: CFGやDFGから）
# features = [
#     "connected_components", "loop_statements", "conditional_statements",
#     "cycles", "paths", "cyclomatic_complexity",
#     "variable_count", "total_reads", "total_writes",
#     "max_reads", "max_writes"
# ]

# # 特徴量の重み（表に基づく）
# weights = np.array([1, 1, 1, 1, 1, 1, 0.6, 0.1, 0.1, 0.1, 0.1])

# # 3つのセントロイド（クラスタ中心）
# centroids = np.array([
#     [1, 0, 1, 0, 1, 2, 2, 1, 1, 1, 1],     # シンプルコード
#     [3, 2, 4, 3, 5, 8, 4, 2, 2, 2, 2],     # 複雑なコード
#     [1, 1, 2, 1, 2, 3, 10, 20, 20, 10, 10] # データ依存のコード
# ])

# # サンプルデータ生成（クラスタごとに50個）
# X = []
# y_true = []

# for i, center in enumerate(centroids):
#     samples = np.random.normal(loc=center, scale=0.5, size=(50, len(center)))
#     samples = np.round(samples).astype(int)
#     samples = np.clip(samples, 0, None)
#     X.append(samples)
#     y_true.extend([i] * 50)

# X = np.vstack(X)
# y_true = np.array(y_true)

# # --- 距離関数（重み対応） ---
# def compute_distance(a, b, metric='euclidean', weights=None):
#     if weights is None:
#         weights = np.ones_like(a)

#     if metric == 'euclidean':
#         return np.sqrt(np.sum(weights * (a - b) ** 2))
#     elif metric == 'manhattan':
#         return np.sum(weights * np.abs(a - b))
#     elif metric == 'cosine':
#         a_w = a * np.sqrt(weights)
#         b_w = b * np.sqrt(weights)
#         return cosine_distances([a_w], [b_w])[0][0]
#     else:
#         raise ValueError("Unsupported metric.")

# # --- k-means++ 初期化（重み対応） ---
# def initialize_centroids_kmeans_pp(data, k, metric='euclidean', weights=None):
#     n_samples = data.shape[0]
#     centroids = [data[np.random.choice(n_samples)]]

#     for _ in range(1, k):
#         distances = np.array([
#             min(compute_distance(x, c, metric, weights) for c in centroids)
#             for x in data
#         ])
#         probs = distances ** 2
#         probs /= probs.sum()
#         next_centroid = data[np.random.choice(n_samples, p=probs)]
#         centroids.append(next_centroid)

#     return np.array(centroids)

# # --- クラスタリング本体 ---
# def custom_clustering_algorithm(data, k=3, metric='euclidean', weights=None, max_iter=100):
#     centroids = initialize_centroids_kmeans_pp(data, k, metric, weights)
#     counts = np.zeros(k)
#     labels = np.full(data.shape[0], -1)

#     for _ in range(max_iter):
#         changed = False
#         for i, x in enumerate(data):
#             # もっとも近いセントロイドを探す
#             min_dist = float('inf')
#             min_idx = 0
#             for idx, c in enumerate(centroids):
#                 d = compute_distance(x, c, metric, weights)
#                 if d < min_dist:
#                     min_dist = d
#                     min_idx = idx

#             if labels[i] != min_idx:
#                 changed = True
#                 labels[i] = min_idx

#             # セントロイド更新（オンライン平均）
#             counts[min_idx] += 1
#             eta = 1 / counts[min_idx]
#             centroids[min_idx] = centroids[min_idx] + eta * (x - centroids[min_idx])

#         if not changed:
#             break

#     return labels, centroids

# # --- 実行と可視化 ---
# labels, centroids_result = custom_clustering_algorithm(
#     X, k=3, metric='euclidean', weights=weights
# )

# # 可視化（PCA 2次元）
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X)
# centroids_2d = pca.transform(centroids_result)

# plt.figure(figsize=(8, 6))
# for i in range(3):
#     plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], label=f"Cluster {i}")
# plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='x', s=100, label='Centroids')
# plt.title("Weighted Clustering on Code Features")
# plt.legend()
# plt.grid(True)
# plt.show()
# print("Centroids:\n", centroids_result)



# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from sklearn.datasets import make_blobs # make_blobsを再利用

# # --- 1. 特徴量の重みを定義 ---
# # Table 1 の特徴量と重みに基づいて、重みの配列を定義します。
# # 11個の特徴量に対応する11個の重みです。
# # 順番はTable 1のFeature列の並びに対応すると仮定します。
# FEATURE_WEIGHTS = np.array([
#     1.0, # connected_components
#     1.0, # loop_statements
#     1.0, # conditional_statements
#     1.0, # cycles
#     1.0, # paths
#     1.0, # cyclomatic_complexity
#     0.6, # variable_count
#     0.1, # total_reads
#     0.1, # total_writes
#     0.1, # max_reads
#     0.1  # max_writes
# ])

# # --- 距離関数の改良 (重み付きユークリッド距離) ---
# # この関数は、ベクトルcとベクトルsに対して、各要素に対応する重みweightsを考慮して距離を計算します。
# def dist(c, s, metric='euclidean', weights=None):
#     if metric == 'euclidean':
#         if weights is None:
#             return np.linalg.norm(c - s)
#         else:
#             # 重み付きユークリッド距離
#             # (x_i - y_i)^2 に重み w_i を掛けて合計し、ルートを取る
#             return np.sqrt(np.sum(weights * (c - s)**2))
#     elif metric == 'manhattan':
#         if weights is None:
#             return np.sum(np.abs(c - s))
#         else:
#             # 重み付きマンハッタン距離
#             return np.sum(weights * np.abs(c - s))
#     elif metric == 'cosine':
#         # コサイン距離に重みを適用するのは複雑になるため、ここでは重みなしとする
#         # 必要であれば、scipy.spatial.distance.cosine を使うなど検討
#         return cosine_distances([c], [s])[0][0]
#     else:
#         raise ValueError(f"未知の距離関数です: {metric}")

# # --- 既存のK-means関連関数（変更なし） ---
# def update_centroid(c, s, n):
#     return c + (1 / n) * (s - c)

# def initialize_centroids(X_data, k):
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
#     kmeans.fit(X_data)
#     return kmeans.cluster_centers_

# # clustering_algorithm 関数に weights 引数を追加
# def clustering_algorithm(X_stream, k, is_correct_fn, metric='euclidean', initial_X_data_for_kmeans_init=None, weights=None):
#     if initial_X_data_for_kmeans_init is None:
#         raise ValueError("initial_X_data_for_kmeans_init must be provided for KMeans++ initialization.")

#     C = initialize_centroids(initial_X_data_for_kmeans_init, k)
#     N = np.zeros(k)
#     centroid_history = [C.copy()]

#     for S in X_stream:
#         # 距離関数に重みを渡す
#         dists = [dist(c, S, metric, weights=weights) for c in C]
#         min_c = np.argmin(dists)
#         N[min_c] += 1

#         if is_correct_fn(S, min_c):
#             C[min_c] = update_centroid(C[min_c], S, N[min_c])
#         centroid_history.append(C.copy())

#     return C, N, centroid_history

# def is_correct_fn_factory(true_centers):
#     def is_correct(S, min_c):
#         # 正解判定の距離計算も、もし重み付きにしたい場合はdist関数を呼び出すように変更
#         # ここでは単純なユークリッド距離（重みなし）で判定している
#         dists = [np.linalg.norm(c - S) for c in true_centers]
#         correct_cluster = np.argmin(dists)
#         return min_c == correct_cluster
#     return is_correct

# # --- メイン関数 ---
# def main(k_clusters: int = 3, n_features: int = 11): # n_featuresを引数に追加し、デフォルトを11に
#     n_samples = 300

#     # make_blobs を使用してデータを生成
#     # n_features を引数として渡す
#     X, y_true = make_blobs(n_samples=n_samples,
#                            centers=k_clusters,
#                            n_features=n_features, # ここで特徴量数を指定
#                            cluster_std=1.0,
#                            random_state=42)

#     # true_centers は make_blobs が内部で生成するクラスタの中心を使う
#     # make_blobsのcentersは指定したクラスタの平均値であるため、別途計算は不要
#     # ただし、make_blobsはtrue_centersを直接返さないため、y_trueから計算する
#     true_centers = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])

#     # クラスタリング実行（セントロイド履歴付き）
#     C_final, N, history = clustering_algorithm(
#         X_stream=iter(X),
#         k=k_clusters,
#         is_correct_fn=is_correct_fn_factory(true_centers),
#         metric='euclidean', # 重み付きユークリッド距離を使用
#         initial_X_data_for_kmeans_init=X,
#         # ここで定義した FEATURE_WEIGHTS を距離関数に渡す
#         # 注意: FEATURE_WEIGHTS の次元数と n_features は一致している必要があります。
#         weights=FEATURE_WEIGHTS
#     )

#     # アニメーション用可視化
#     # make_blobsのn_featuresが2の場合のみ、直接プロットが可能
#     # それ以外の場合は、次元削減（PCAなど）が必要
#     if n_features == 2:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True data')
#         centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label='Centroids')

#         # 軸ラベルは一般的なFeature 1, Feature 2とする
#         ax.set_xlabel("Feature 1")
#         ax.set_ylabel("Feature 2")

#         def update(frame):
#             current_frame = min(frame, len(history) - 1)
#             centers = history[current_frame]
#             centers_scatter.set_offsets(centers) # 2次元なのでそのままオフセット
#             ax.set_title(f"ステップ {current_frame + 1} / {len(history)} (k={k_clusters})")
#             return centers_scatter,

#         ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True, repeat=False)
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     else:
#         print(f"\nデータは{n_features}次元のため、直接的な2Dプロットはできません。")
#         print("可視化するには、PCAなどの次元削減手法を適用する必要があります。")

#     print(f"最終的なセントロイド（k={k_clusters}）:\n", np.round(C_final, 2))


# if __name__ == '__main__':
#     # 例: k=3, 特徴量数=11 で実行（Table 1の重みに対応）
#     main(k_clusters=3, n_features=11)

#     # 可視化を試したい場合は、n_features=2 に変更して実行
#     # main(k_clusters=3, n_features=2)

#     # kを変えて試すことも可能（例: k=4, n_features=11）
#     # main(k_clusters=4, n_features=11)



# これは特徴量と重みを考慮したkmeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA # PCAをインポート

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

# --- クラスタリングアルゴリズム（履歴記録つき） ---
def clustering_algorithm(X_stream, k, is_correct_fn, metric='euclidean', initial_X_data_for_kmeans_init=None, weights=None):
    if initial_X_data_for_kmeans_init is None:
        raise ValueError("initial_X_data_for_kmeans_init must be provided for KMeans++ initialization.")

    C = initialize_centroids(initial_X_data_for_kmeans_init, k)
    N = np.zeros(k)
    centroid_history = [C.copy()]

    for S in X_stream:
        dists = [dist(c, S, metric, weights=weights) for c in C]
        min_c = np.argmin(dists)
        N[min_c] += 1

        if is_correct_fn(S, min_c):
            C[min_c] = update_centroid(C[min_c], S, N[min_c])
        centroid_history.append(C.copy())

    return C, N, centroid_history

# --- 正解判定（教師あり） ---
def is_correct_fn_factory(true_centers):
    def is_correct(S, min_c):
        dists = [np.linalg.norm(c - S) for c in true_centers]
        correct_cluster = np.argmin(dists)
        return min_c == correct_cluster
    return is_correct

# --- メイン関数 ---
def main(k_clusters: int = 3, n_features: int = 11):
    n_samples = 300

    # make_blobs を使用してデータを生成
    X, y_true = make_blobs(n_samples=n_samples,
                           centers=k_clusters,
                           n_features=n_features,
                           cluster_std=1.0,
                           random_state=42)

    true_centers = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])

    # クラスタリング実行（セントロイド履歴付き）
    C_final, N, history = clustering_algorithm(
        X_stream=iter(X),
        k=k_clusters,
        is_correct_fn=is_correct_fn_factory(true_centers),
        metric='euclidean',
        initial_X_data_for_kmeans_init=X,
        weights=FEATURE_WEIGHTS
    )

    # --- PCAによる次元削減と可視化 ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 履歴内の各セントロイドも2次元に変換
    history_2d = [pca.transform(h) for h in history]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 初期データをPCA変換したものでプロット
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True data')

    # セントロイドのプロットは最初は空で、アニメーションで更新
    centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label='Centroids')

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Weighted Clustering (PCA 2D Projection, k={k_clusters})")
    plt.legend()
    plt.grid(True)

    def update(frame):
        current_frame = min(frame, len(history_2d) - 1)
        centers_2d = history_2d[current_frame] # 2次元に変換されたセントロイドを取得
        centers_scatter.set_offsets(centers_2d)
        ax.set_title(f"ステップ {current_frame + 1} / {len(history_2d)} (k={k_clusters})")
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
    plt.show()

    print(f"最終的なセントロイド（k={k_clusters}）:\n", np.round(C_final, 2))
    print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")


if __name__ == '__main__':
    # Table 1 の重みと対応する11次元データで実行し、PCAで可視化
    main(k_clusters=3, n_features=11)

    # 2次元データで実行したい場合は以下
    # main(k_clusters=3, n_features=2)
