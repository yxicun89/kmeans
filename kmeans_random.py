# これはノイズないけどまばらな分布のkmeans
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

    if n_features == 2:
        centers_for_blobs = [
            [-10, 10],   # クラスタ1
            [0, -10],    # クラスタ2
            [10, 10]     # クラスタ3
        ]
        if k_clusters > 3:
            centers_for_blobs.append([-10, -10])
        if k_clusters > 4:
            centers_for_blobs.append([0, 0])

        # 指定された k_clusters の数だけ centers を使用
        centers_for_blobs = centers_for_blobs[:k_clusters]

        X, y_true = make_blobs(n_samples=n_samples,
                               centers=centers_for_blobs,
                               n_features=n_features,
                               cluster_std=5.0,
                               random_state=42)
    else:
        random_centers = np.random.uniform(low=-20, high=20, size=(k_clusters, n_features))

        X, y_true = make_blobs(n_samples=n_samples,
                               centers=random_centers,
                               n_features=n_features,
                               cluster_std=7.0,
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
    history_2d = [pca.transform(h) for h in history]

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True data')
    centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label='Centroids')

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Weighted Clustering (PCA 2D Projection, k={k_clusters}) - Sparse Data") # タイトルにSparse Dataを追加
    plt.legend()
    plt.grid(True)

    def update(frame):
        current_frame = min(frame, len(history_2d) - 1)
        centers_2d = history_2d[current_frame]
        centers_scatter.set_offsets(centers_2d)
        ax.set_title(f"ステップ {current_frame + 1} / {len(history_2d)} (k={k_clusters}) - Sparse Data")
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
    plt.show()

    print(f"最終的なセントロイド（k={k_clusters}）:\n", np.round(C_final, 2))
    print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")


if __name__ == '__main__':
    # 例: k=3, 特徴量数=11 で実行（まばらな分布）
    main(k_clusters=3, n_features=11)

    # 2次元でまばらな分布を確認したい場合
    # main(k_clusters=3, n_features=2)

    # kを変えて試す場合（例: k=4, n_features=11）
    # main(k_clusters=4, n_features=11)


# これはノイズ+すべてのkが統一のkmeans
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# # from sklearn.datasets import make_blobs # make_blobs は使用しない
# from sklearn.decomposition import PCA

# # --- 特徴量の重みを定義 ---
# # Table 1 の特徴量と重みに基づいて、重みの配列を定義します。
# # 11個の特徴量に対応する11個の重みです。
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

# # --- 距離関数（重み付きユークリッド距離） ---
# def dist(c, s, metric='euclidean', weights=None):
#     if metric == 'euclidean':
#         if weights is None:
#             return np.linalg.norm(c - s)
#         else:
#             return np.sqrt(np.sum(weights * (c - s)**2))
#     elif metric == 'manhattan':
#         if weights is None:
#             return np.sum(np.abs(c - s))
#         else:
#             return np.sum(weights * np.abs(c - s))
#     elif metric == 'cosine':
#         if weights is None:
#             return cosine_distances([c], [s])[0][0]
#         else:
#             c_w = c * np.sqrt(weights)
#             s_w = s * np.sqrt(weights)
#             return cosine_distances([c_w], [s_w])[0][0]
#     else:
#         raise ValueError(f"未知の距離関数です: {metric}")

# # --- セントロイド更新 ---
# def update_centroid(c, s, n):
#     return c + (1 / n) * (s - c)

# # --- k-means++ 初期化 ---
# def initialize_centroids(X_data, k):
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
#     kmeans.fit(X_data)
#     return kmeans.cluster_centers_

# # --- クラスタリングアルゴリズム（履歴記録つき） ---
# def clustering_algorithm(X_stream, k, is_correct_fn, metric='euclidean', initial_X_data_for_kmeans_init=None, weights=None):
#     if initial_X_data_for_kmeans_init is None:
#         raise ValueError("initial_X_data_for_kmeans_init must be provided for KMeans++ initialization.")

#     C = initialize_centroids(initial_X_data_for_kmeans_init, k)
#     N = np.zeros(k)
#     centroid_history = [C.copy()]

#     for S in X_stream:
#         dists = [dist(c, S, metric, weights=weights) for c in C]
#         min_c = np.argmin(dists)
#         N[min_c] += 1

#         if is_correct_fn(S, min_c):
#             C[min_c] = update_centroid(C[min_c], S, N[min_c])
#         centroid_history.append(C.copy())

#     return C, N, centroid_history

# # --- 正解判定（教師あり） ---
# def is_correct_fn_factory(true_centers):
#     def is_correct(S, min_c):
#         dists = [np.linalg.norm(c - S) for c in true_centers]
#         correct_cluster = np.argmin(dists)
#         return min_c == correct_cluster
#     return is_correct

# # --- まばらなクラスタデータ生成関数 (再掲) ---
# def generate_sparse_clusters(n_per_cluster=40, noise_ratio=0.2, seed=42):
#     np.random.seed(seed)
#     base_centers = np.array([
#         [1, 0, 1, 0, 1, 2, 2, 1, 1, 1, 1],
#         [3, 2, 4, 3, 5, 8, 4, 2, 2, 2, 2],
#         [1, 1, 2, 1, 2, 3, 10, 20, 20, 10, 10]
#     ])

#     data = []
#     labels = []
#     for i, center in enumerate(base_centers):
#         samples = np.random.normal(loc=center, scale=1.5, size=(n_per_cluster, len(center)))
#         samples = np.round(samples).astype(int)
#         samples = np.clip(samples, 0, None)
#         data.append(samples)
#         labels.extend([i] * n_per_cluster)

#     n_noise = int(len(labels) * noise_ratio)
#     # ノイズの範囲を特徴量の一般的な範囲に合わせて調整することも検討
#     # ここでは元のコードと同様に0-25の範囲
#     noise = np.random.randint(0, 25, size=(n_noise, base_centers.shape[1]))

#     data.append(noise)
#     labels.extend([-1] * n_noise) # ノイズデータはラベル-1

#     X_sparse = np.vstack(data)
#     y_sparse = np.array(labels)

#     # K-means++初期化のために、元の真のクラスタ中心を返す
#     # 注意: y_sparse に -1 のノイズラベルが含まれるため、
#     # true_centers は base_centers を直接使う方が意図に合うことが多い
#     return X_sparse, y_sparse, base_centers # base_centersも返すように変更

# # --- メイン関数 ---
# def main(k_clusters: int = 3): # n_features は generate_sparse_clusters で固定されるため引数から削除
#     # generate_sparse_clusters を使用してデータを生成
#     # k_clusters は generate_sparse_clusters のデフォルト（3つのクラスタ）と一致させる
#     # もし k_clusters を generate_sparse_clusters に渡したい場合は、generate_sparse_clustersを変更
#     if k_clusters != 3:
#         print(f"Warning: generate_sparse_clusters creates 3 clusters. k_clusters={k_clusters} might lead to mismatch with true labels.")
#         print("Continuing with k_clusters as specified, but keep in mind data has 3 inherent clusters + noise.")
#         # ここで、generate_sparse_clusters の base_centers を k_clusters に合わせて動的に生成するなどの改良も可能

#     X, y_true, true_centers_for_eval = generate_sparse_clusters(
#         n_per_cluster=40, # 各クラスタのデータ点数
#         noise_ratio=0.2,  # ノイズの割合
#         seed=42
#     )

#     # generate_sparse_clusters は11次元データを生成するため、n_featuresは常に11
#     n_features = X.shape[1]

#     # クラスタリング実行（セントロイド履歴付き）
#     C_final, N, history = clustering_algorithm(
#         X_stream=iter(X),
#         k=k_clusters,
#         # 正解判定には generate_sparse_clusters が返す base_centers を使う
#         is_correct_fn=is_correct_fn_factory(true_centers_for_eval),
#         metric='euclidean', # 重み付きユークリッド距離を使用
#         initial_X_data_for_kmeans_init=X,
#         weights=FEATURE_WEIGHTS
#     )

#     # --- PCAによる次元削減と可視化 ---
#     # 常にPCAで2次元に削減して可視化
#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X) # データXを2次元に変換

#     # 履歴内の各セントロイドも2次元に変換
#     history_2d = [pca.transform(h) for h in history]

#     # true_centers_for_eval もPCA変換してプロットできるようにする
#     true_centers_2d = pca.transform(true_centers_for_eval)

#     fig, ax = plt.subplots(figsize=(8, 6))

#     # y_true には -1 のノイズラベルが含まれるので、それらを区別してプロット
#     # -1 のラベルを持つノイズ点は灰色などで表示
#     unique_labels = np.unique(y_true)
#     for label in unique_labels:
#         if label == -1:
#             ax.scatter(X_2d[y_true == label, 0], X_2d[y_true == label, 1],
#                        c='gray', alpha=0.3, label='Noise')
#         else:
#             ax.scatter(X_2d[y_true == label, 0], X_2d[y_true == label, 1],
#                        label=f"True Cluster {label}", alpha=0.5)

#     # 真のクラスタ中心をプロット
#     ax.scatter(true_centers_2d[:, 0], true_centers_2d[:, 1],
#                c='blue', s=200, marker='o', edgecolors='black', label='True Centers (Base)')

#     # セントロイドのプロットは最初は空で、アニメーションで更新
#     centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label='Calculated Centroids')

#     ax.set_xlabel("Principal Component 1")
#     ax.set_ylabel("Principal Component 2")
#     ax.set_title(f"Weighted Clustering (PCA 2D Projection, k={k_clusters}) - Sparse Data with Noise")
#     plt.legend()
#     plt.grid(True)

#     def update(frame):
#         current_frame = min(frame, len(history_2d) - 1)
#         centers_2d = history_2d[current_frame] # 2次元に変換されたセントロイドを取得
#         centers_scatter.set_offsets(centers_2d)
#         ax.set_title(f"ステップ {current_frame + 1} / {len(history_2d)} (k={k_clusters}) - Sparse Data with Noise")
#         return centers_scatter,

#     ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
#     plt.show()

#     print(f"最終的なセントロイド（k={k_clusters}）:\n", np.round(C_final, 2))
#     print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")
#     print("ノイズ点はラベル -1 として灰色で表示されます。")


# if __name__ == '__main__':
#     # generate_sparse_clusters は常に11次元データを生成します。
#     # k_clusters は generate_sparse_clusters が生成するクラスタ数（3）と一致させるのが一般的です。
#     main(k_clusters=3)

#     # k_clusters=4 などにすると、クラスタ数が合わない警告が出ますが、実行は可能です。
#     # その場合、K-meansは4つのクラスタを見つけようとします。
#     # main(k_clusters=4)




# これはセントロイドだけ増やしたkemans
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from sklearn.decomposition import PCA

# # --- 特徴量の重みを定義 ---
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

# # --- 距離関数（重み付きユークリッド距離） ---
# def dist(c, s, metric='euclidean', weights=None):
#     if metric == 'euclidean':
#         if weights is None:
#             return np.linalg.norm(c - s)
#         else:
#             return np.sqrt(np.sum(weights * (c - s)**2))
#     elif metric == 'manhattan':
#         if weights is None:
#             return np.sum(np.abs(c - s))
#         else:
#             return np.sum(weights * np.abs(c - s))
#     elif metric == 'cosine':
#         if weights is None:
#             return cosine_distances([c], [s])[0][0]
#         else:
#             c_w = c * np.sqrt(weights)
#             s_w = s * np.sqrt(weights)
#             return cosine_distances([c_w], [s_w])[0][0]
#     else:
#         raise ValueError(f"未知の距離関数です: {metric}")

# # --- セントロイド更新 ---
# def update_centroid(c, s, n):
#     return c + (1 / n) * (s - c)

# # --- k-means++ 初期化 ---
# def initialize_centroids(X_data, k):
#     # ここで KMeans が k 個のセントロイドを初期化します。
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
#     kmeans.fit(X_data)
#     return kmeans.cluster_centers_

# # --- クラスタリングアルゴリズム（履歴記録つき） ---
# def clustering_algorithm(X_stream, k, is_correct_fn, metric='euclidean', initial_X_data_for_kmeans_init=None, weights=None):
#     # ここで k がクラスタ数として使われます。
#     C = initialize_centroids(initial_X_data_for_kmeans_init, k)
#     N = np.zeros(k) # Nもkの数に合わせる
#     centroid_history = [C.copy()]

#     for S in X_stream:
#         dists = [dist(c, S, metric, weights=weights) for c in C]
#         min_c = np.argmin(dists)
#         N[min_c] += 1

#         if is_correct_fn(S, min_c):
#             C[min_c] = update_centroid(C[min_c], S, N[min_c])
#         centroid_history.append(C.copy())

#     return C, N, centroid_history

# # --- 正解判定（教師あり） ---
# def is_correct_fn_factory(true_centers):
#     def is_correct(S, min_c):
#         dists = [np.linalg.norm(c - S) for c in true_centers]
#         correct_cluster = np.argmin(dists)
#         return min_c == correct_cluster
#     return is_correct

# # --- まばらなクラスタデータ生成関数 ---
# def generate_sparse_clusters(n_per_cluster=40, noise_ratio=0.2, seed=42):
#     np.random.seed(seed)
#     # この base_centers は常に3つのクラスタを生成します。
#     # K-meansのkの値とは独立しています。
#     base_centers = np.array([
#         [1, 0, 1, 0, 1, 2, 2, 1, 1, 1, 1],
#         [3, 2, 4, 3, 5, 8, 4, 2, 2, 2, 2],
#         [1, 1, 2, 1, 2, 3, 10, 20, 20, 10, 10]
#     ])

#     data = []
#     labels = []
#     for i, center in enumerate(base_centers):
#         samples = np.random.normal(loc=center, scale=1.5, size=(n_per_cluster, len(center)))
#         samples = np.round(samples).astype(int)
#         samples = np.clip(samples, 0, None)
#         data.append(samples)
#         labels.extend([i] * n_per_cluster)

#     n_noise = int(len(labels) * noise_ratio)
#     noise = np.random.randint(0, 25, size=(n_noise, base_centers.shape[1]))

#     data.append(noise)
#     labels.extend([-1] * n_noise)

#     X_sparse = np.vstack(data)
#     y_sparse = np.array(labels)

#     return X_sparse, y_sparse, base_centers

# # --- メイン関数 ---
# def main(k_clusters: int = 3): # K-meansが使用するクラスタ数
#     # generate_sparse_clusters は常に3つの「真の」クラスタとノイズを生成します。
#     X, y_true, true_centers_for_eval = generate_sparse_clusters(
#         n_per_cluster=40,
#         noise_ratio=0.2,
#         seed=42
#     )

#     n_features = X.shape[1]

#     # K-meansアルゴリズムに、ここで指定した k_clusters を渡します。
#     # generate_sparse_clusters の内部クラスタ数（3）とは独立しています。
#     C_final, N, history = clustering_algorithm(
#         X_stream=iter(X),
#         k=k_clusters, # ここで K-means のクラスタ数を指定
#         is_correct_fn=is_correct_fn_factory(true_centers_for_eval),
#         metric='euclidean',
#         initial_X_data_for_kmeans_init=X,
#         weights=FEATURE_WEIGHTS
#     )

#     # --- PCAによる次元削減と可視化 ---
#     pca = PCA(n_components=2)
#     X_2d = pca.fit_transform(X)
#     history_2d = [pca.transform(h) for h in history]
#     true_centers_2d = pca.transform(true_centers_for_eval)

#     fig, ax = plt.subplots(figsize=(8, 6))

#     unique_labels = np.unique(y_true)
#     for label in unique_labels:
#         if label == -1:
#             ax.scatter(X_2d[y_true == label, 0], X_2d[y_true == label, 1],
#                        c='gray', alpha=0.3, label='Noise')
#         else:
#             ax.scatter(X_2d[y_true == label, 0], X_2d[y_true == label, 1],
#                        label=f"True Cluster {label}", alpha=0.5)

#     ax.scatter(true_centers_2d[:, 0], true_centers_2d[:, 1],
#                c='blue', s=200, marker='o', edgecolors='black', label='True Centers (Base)')

#     # セントロイドのプロットは赤色のXマーカー
#     centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label=f'Calculated Centroids (k={k_clusters})')

#     ax.set_xlabel("Principal Component 1")
#     ax.set_ylabel("Principal Component 2")
#     ax.set_title(f"Weighted Clustering (PCA 2D Projection) - K-means k={k_clusters} on Sparse Data with Noise") # タイトルにkを表示
#     plt.legend()
#     plt.grid(True)

#     def update(frame):
#         current_frame = min(frame, len(history_2d) - 1)
#         centers_2d = history_2d[current_frame]
#         centers_scatter.set_offsets(centers_2d)
#         ax.set_title(f"ステップ {current_frame + 1} / {len(history_2d)} (K-means k={k_clusters}) - Sparse Data with Noise")
#         return centers_scatter,

#     ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
#     plt.show()

#     print(f"最終的なセントロイド（K-meansのk={k_clusters}）:\n", np.round(C_final, 2))
#     print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")
#     print("ノイズ点はラベル -1 として灰色で表示されます。")
#     print(f"データは3つの真のクラスタとノイズから構成されています。K-meansは{k_clusters}個のクラスタを見つけようとしました。")


# if __name__ == '__main__':
#     # データ生成は常に3つの真のクラスタとノイズを含みます。
#     # K-meansが4つのクラスタを見つけようとすることで、ノイズをクラスタとして捉えるか試す
#     print("--- 実行: K-means k=4 でクラスタリング ---")
#     main(k_clusters=4)

#     # K-meansが3つのクラスタを見つけようとした場合
#     # print("\n--- 実行: K-means k=3 でクラスタリング ---")
#     # main(k_clusters=3)

#     # K-meansが5つのクラスタを見つけようとした場合
#     # print("\n--- 実行: K-means k=5 でクラスタリング ---")
#     # main(k_clusters=5)
