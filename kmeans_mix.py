import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, load_wine
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
        # make_blobs, iris, wine などはユークリッド距離で真の中心を判断
        dists = [np.linalg.norm(c - S) for c in true_centers]
        correct_cluster = np.argmin(dists)
        return min_c == correct_cluster
    return is_correct

# --- データセット作成関数 ---
def create_dataset(dataset_name: str, n_samples: int = 300):
    """
    指定された名前のデータセットを生成し、特徴量データXと真のラベルy_trueを返す。
    また、データセットの特性に合わせたk_clustersとn_featuresも返す。
    """
    if dataset_name == 'blobs':
        k_clusters = 3
        n_features = 11 # 以前のFEATURE_WEIGHTSに合わせる
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=k_clusters,
                               n_features=n_features,
                               cluster_std=1.0,
                               random_state=42)
    elif dataset_name == 'circles':
        k_clusters = 2 # 同心円は通常2クラス
        n_features = 2
        X, y_true = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=42)
    elif dataset_name == 'moons':
        k_clusters = 2 # 三日月は通常2クラス
        n_features = 2
        X, y_true = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    elif dataset_name == 'iris':
        iris = load_iris()
        X, y_true = iris.data, iris.target
        k_clusters = len(np.unique(y_true)) # 3クラス
        n_features = X.shape[1] # 4特徴量
    elif dataset_name == 'wine':
        wine = load_wine()
        X, y_true = wine.data, wine.target
        k_clusters = len(np.unique(y_true)) # 3クラス
        n_features = X.shape[1] # 13特徴量
    else:
        raise ValueError(f"不明なデータセット名です: {dataset_name}")
    
    # 真のセントロイドを計算
    true_centers = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])
    
    return X, y_true, k_clusters, n_features, true_centers

# --- メイン関数 ---
def main(dataset_name: str):
    # データセットの生成
    X, y_true, k_clusters, n_features, true_centers = create_dataset(dataset_name)

    # PCAを適用するかどうかのフラグ (2次元データの場合は不要、高次元の場合に適用)
    apply_pca = n_features > 2

    # クラスタリング実行（セントロイド履歴付き）
    C_final, N, history = clustering_algorithm(
        X_stream=iter(X),
        k=k_clusters,
        is_correct_fn=is_correct_fn_factory(true_centers),
        metric='euclidean', # ここは必要に応じて変更
        initial_X_data_for_kmeans_init=X,
        weights=FEATURE_WEIGHTS if dataset_name == 'blobs' else None # blobs以外では重みを適用しない
    )

    # --- PCAによる次元削減と可視化 ---
    if apply_pca:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        history_2d = [pca.transform(h) for h in history]
        plot_title_suffix = f" (PCA 2D Projection, k={k_clusters})"
    else:
        # 2次元データはそのまま使用
        X_2d = X
        history_2d = history
        plot_title_suffix = f" (k={k_clusters})"
        # 2次元データの場合、FEATURE_WEIGHTSの次元が合わないため、警告を出すか調整が必要です
        if n_features != len(FEATURE_WEIGHTS) and dataset_name != 'blobs':
             print(f"Warning: FEATURE_WEIGHTS is defined for {len(FEATURE_WEIGHTS)} features, but {dataset_name} has {n_features} features. Weights are currently not applied for non-'blobs' datasets.")


    fig, ax = plt.subplots(figsize=(8, 6))

    # 初期データをPCA変換したものでプロット
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True data')

    # セントロイドのプロットは最初は空で、アニメーションで更新
    centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label='Centroids')

    ax.set_xlabel("Principal Component 1" if apply_pca else "Feature 1")
    ax.set_ylabel("Principal Component 2" if apply_pca else "Feature 2")
    ax.set_title(f"Clustering on {dataset_name.capitalize()} Dataset" + plot_title_suffix)
    plt.legend()
    plt.grid(True)

    def update(frame):
        current_frame = min(frame, len(history_2d) - 1)
        centers_2d = history_2d[current_frame] # 2次元に変換されたセントロイドを取得
        centers_scatter.set_offsets(centers_2d)
        ax.set_title(f"Clustering on {dataset_name.capitalize()} Dataset - ステップ {current_frame + 1} / {len(history_2d)}" + plot_title_suffix)
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
    plt.show()

    print(f"--- {dataset_name.capitalize()} Dataset Results (k={k_clusters}) ---")
    print(f"最終的なセントロイド:\n", np.round(C_final, 2))
    if apply_pca:
        print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")
    print("-" * 50)


if __name__ == '__main__':
    print("異なるデータセットでのK-meansクラスタリングを開始します。")

    # 理想的なデータセット
    main(dataset_name='blobs')
    main(dataset_name='circles')
    main(dataset_name='moons')

    # 現実的なデータセット
    main(dataset_name='iris')
    main(dataset_name='wine')

    print("\n全てのデータセットでの動作確認が完了しました。")
    print("各データセットの結果ウィンドウを閉じると、次のデータセットが表示されます。")
    print("Warning: FEATURE_WEIGHTSは'blobs'データセットのみに適用されています。")
    print("他のデータセットに適用するには、各データセットのn_featuresに合わせてFEATURE_WEIGHTSの次元を調整する必要があります。")