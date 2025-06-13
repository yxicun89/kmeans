import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, load_wine
from sklearn.decomposition import PCA # PCAをインポート

# --- 特徴量の重みを定義 ---
# connected_components, loop_statements, conditional_statements, cycles, paths, cyclomatic_complexity
# variable_count, total_reads, total_writes, max_reads, max_writes に対応
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

# --- 距離関数（重み付きユークリッド距離、マンハッタン距離、コサイン距離） ---
def dist(c, s, metric='euclidean', weights=None):
    """
    2つの点 c と s 間の距離を計算します。
    重みが指定されている場合、重み付き距離を計算します。
    """
    if metric == 'euclidean':
        if weights is None:
            return np.linalg.norm(c - s)
        else:
            # 重み付きユークリッド距離: sqrt(sum(w_i * (c_i - s_i)^2))
            return np.sqrt(np.sum(weights * (c - s)**2))
    elif metric == 'manhattan':
        if weights is None:
            return np.sum(np.abs(c - s))
        else:
            # 重み付きマンハッタン距離: sum(w_i * |c_i - s_i|)
            return np.sum(weights * np.abs(c - s))
    elif metric == 'cosine':
        if weights is None:
            return cosine_distances([c], [s])[0][0]
        else:
            # 重み付きコサイン距離: 重みをsqrt(w_i)で特徴量に適用してからコサイン距離を計算
            c_w = c * np.sqrt(weights)
            s_w = s * np.sqrt(weights)
            return cosine_distances([c_w], [s_w])[0][0]
    else:
        raise ValueError(f"未知の距離関数です: {metric}")

# --- K-means++ 初期化 ---
def initialize_centroids(X_data, k):
    """
    K-means++アルゴリズムを使用してセントロイドを初期化します。
    """
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_data)
    return kmeans.cluster_centers_

# --- 一般的なK-meansクラスタリングアルゴリズム（履歴記録つき） ---
def general_kmeans_algorithm(X_data, k, metric='euclidean', weights=None, max_iterations=100):
    """
    一般的なK-meansクラスタリングアルゴリズムを実行し、セントロイドの履歴を記録します。
    
    Args:
        X_data (np.ndarray): クラスタリング対象のデータ。
        k (int): クラスターの数。
        metric (str): 距離関数の種類 ('euclidean', 'manhattan', 'cosine')。
        weights (np.ndarray, optional): 各特徴量に適用する重み。
        max_iterations (int): 最大イテレーション数。

    Returns:
        tuple: 最終的なセントロイド、最終的なラベル、セントロイドの履歴。
    """
    C = initialize_centroids(X_data, k)
    centroid_history = [C.copy()]

    for iteration in range(max_iterations):
        # ステップ 1: 各データポイントを最も近いセントロイドに割り当てる
        labels = np.zeros(len(X_data), dtype=int)
        for i, S in enumerate(X_data):
            dists = [dist(c, S, metric, weights=weights) for c in C]
            labels[i] = np.argmin(dists)

        # ステップ 2: 新しいクラスター割り当てに基づいてセントロイドを更新
        new_C = np.zeros((k, X_data.shape[1]))
        for i in range(k):
            points_in_cluster = X_data[labels == i]
            if len(points_in_cluster) > 0:
                new_C[i] = np.mean(points_in_cluster, axis=0)
            else:
                # クラスターが空になった場合、データ全体の範囲内でランダムに再初期化する
                min_val = np.min(X_data, axis=0)
                max_val = np.max(X_data, axis=0)
                new_C[i] = np.random.uniform(min_val, max_val, X_data.shape[1])
                
        # 収束判定: セントロイドがほとんど変化しなくなったら停止
        if np.allclose(C, new_C):
            # print(f"収束しました (Iteration {iteration + 1})") # デバッグ用
            break
        
        C = new_C
        centroid_history.append(C.copy())
    
    # 最終的なラベル付け (アニメーション終了後の状態を表示するため)
    final_labels = np.zeros(len(X_data), dtype=int)
    for i, S in enumerate(X_data):
        dists = [dist(c, S, metric, weights=weights) for c in C]
        final_labels[i] = np.argmin(dists)

    return C, final_labels, centroid_history

# --- データセット作成関数 ---
def create_dataset(dataset_name: str, n_samples: int = 300):
    """
    指定された名前のデータセットを生成し、特徴量データXと真のラベルy_trueを返す。
    また、データセットの特性に合わせたk_clustersとn_featuresも返す。
    真のセントロイドは、該当する場合に計算される。
    """
    if dataset_name == 'blobs':
        k_clusters = 3
        n_features = 11 # FEATURE_WEIGHTSに合わせる
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
        X, y_true = wine.data, wine.target # <-- ここを修正しました
        k_clusters = len(np.unique(y_true)) # 3クラス
        n_features = X.shape[1] # 13特徴量
    elif dataset_name == 'random':
        k_clusters = 3 # K-meansはクラスター数を指定する必要があるため、便宜的に3とする
        n_features = 2 # 可視化のため2次元とする
        X = np.random.rand(n_samples, n_features) * 10 # 0-10の範囲でランダムデータを生成
        y_true = np.zeros(n_samples) # ランダムデータには真のラベルがないため、ダミーを設定
    elif dataset_name == 'clustered_random':
        k_clusters = 3 # クラスターの数
        n_features = 2 # 可視化のため2次元とする
        cluster_std = 1.0 # 各クラスター内の点のばらつき
        
        # 真のクラスター中心を手動で定義
        true_centers = np.array([
            [2, 2],
            [8, 3],
            [5, 8]
        ])
        
        X_list = []
        y_list = []
        # 各クラスター中心の周りにデータを生成
        for i, center in enumerate(true_centers):
            num_points_in_cluster = n_samples // k_clusters
            X_cluster = np.random.normal(loc=center, scale=cluster_std, size=(num_points_in_cluster, n_features))
            X_list.append(X_cluster)
            y_list.append(np.full(num_points_in_cluster, i)) # 真のラベルを割り当てる

        X = np.vstack(X_list)
        y_true = np.hstack(y_list)
    elif dataset_name == 'overlapping_blobs':
        k_clusters = 3
        n_features = 2 # 可視化しやすいように2次元
        n_samples_per_cluster = n_samples // k_clusters
        
        # 非常に近接したクラスター中心
        centers = np.array([
            [0, 0],
            [0.5, 0.5],
            [-0.5, 0.5]
        ])
        
        # 大きな標準偏差 (クラスターが重なり合うように)
        cluster_std = 1.5 
        
        X_list = []
        y_list = []
        for i, center in enumerate(centers):
            X_cluster = np.random.normal(loc=center, scale=cluster_std, size=(n_samples_per_cluster, n_features))
            X_list.append(X_cluster)
            y_list.append(np.full(n_samples_per_cluster, i))
            
        X = np.vstack(X_list)
        y_true = np.hstack(y_list)
        
    else:
        raise ValueError(f"不明なデータセット名です: {dataset_name}")
    
    # 真のセントロイドを計算 (K-meansの教師なし設定では直接利用しないが、可視化や評価の比較用)
    # randomデータには真のクラスターが存在しないため、Noneを設定
    if dataset_name == 'random':
        true_centers = None
    elif dataset_name in ['clustered_random', 'overlapping_blobs']:
        # clustered_random, overlapping_blobsでは既にtrue_centersを定義済み、または後で計算
        if 'true_centers' not in locals(): # create_dataset内で定義されていない場合
            true_centers = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])
        pass
    else:
        true_centers = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])
    
    return X, y_true, k_clusters, n_features, true_centers

# --- メイン関数 ---
def main(dataset_name: str):
    # データセットの生成
    X, y_true, k_clusters, n_features, true_centers = create_dataset(dataset_name)

    # PCAを適用するかどうかのフラグ (2次元データの場合は不要、高次元の場合に適用)
    apply_pca = n_features > 2

    # クラスタリング実行（セントロイド履歴付き）
    C_final, final_labels, history = general_kmeans_algorithm(
        X_data=X,
        k=k_clusters,
        metric='euclidean', # 必要に応じて 'manhattan' や 'cosine' に変更可能
        weights=FEATURE_WEIGHTS if dataset_name == 'blobs' else None # 'blobs'以外では重みを適用しない
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
        # 注意: FEATURE_WEIGHTSの次元がデータセットと異なる場合の警告
        if n_features != len(FEATURE_WEIGHTS) and dataset_name != 'blobs':
             print(f"Warning: FEATURE_WEIGHTS is defined for {len(FEATURE_WEIGHTS)} features, but {dataset_name} has {n_features} features. Weights are currently not applied for non-'blobs' datasets.")


    fig, ax = plt.subplots(figsize=(8, 6))

    # データポイントのプロット
    if dataset_name == 'random':
        # 'random'データセットには真のラベルがないため、K-meansが割り当てたラベルで色分け
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=final_labels, cmap='viridis', alpha=0.5, label='Clustered Data (K-means)')
    else:
        # 他のデータセットでは真のラベル（y_true）で色分け
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True Data (Ground Truth)')
    
    # 真のセントロイドが存在する場合、それもプロット
    if true_centers is not None:
        if apply_pca:
            true_centers_2d = pca.transform(true_centers)
            ax.scatter(true_centers_2d[:, 0], true_centers_2d[:, 1], c='blue', s=200, marker='o', edgecolor='black', label='True Centers (PCA)')
        else:
            ax.scatter(true_centers[:, 0], true_centers[:, 1], c='blue', s=200, marker='o', edgecolor='black', label='True Centers', alpha=0.7)


    # セントロイドのプロットは最初は空で、アニメーションで更新
    centers_scatter = ax.scatter([], [], c='red', s=150, marker='X', label='Centroids')

    ax.set_xlabel("Principal Component 1" if apply_pca else "Feature 1")
    ax.set_ylabel("Principal Component 2" if apply_pca else "Feature 2")
    ax.set_title(f"General K-means on {dataset_name.capitalize()} Dataset" + plot_title_suffix)
    plt.legend()
    plt.grid(True)

    def update(frame):
        current_frame = min(frame, len(history_2d) - 1)
        centers_2d = history_2d[current_frame] # 2次元に変換されたセントロイドを取得
        centers_scatter.set_offsets(centers_2d)
        ax.set_title(f"General K-means on {dataset_name.capitalize()} Dataset - イテレーション {current_frame + 1} / {len(history_2d)}" + plot_title_suffix)
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
    plt.show()

    print(f"--- {dataset_name.capitalize()} Dataset Results (k={k_clusters}) ---")
    print(f"最終的なセントロイド:\n", np.round(C_final, 2))
    if apply_pca:
        print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")
    print("-" * 50)


if __name__ == '__main__':
    print("異なるデータセットでの一般的なK-meansクラスタリングを開始します。")

    # 混ざり合ったクラスターデータセット
    main(dataset_name='overlapping_blobs')
    
    # 理想的なデータセット (make_blobsに近いものを自作)
    main(dataset_name='clustered_random')
    
    # 理想的なデータセット (既存のmake_blobs)
    main(dataset_name='blobs')
    main(dataset_name='circles')
    main(dataset_name='moons')

    # 現実的なデータセット
    main(dataset_name='iris')
    main(dataset_name='wine')

    # 真にランダムなデータセット
    main(dataset_name='random')

    print("\n全てのデータセットでの動作確認が完了しました。")
    print("各データセットの結果ウィンドウを閉じると、次のデータセットが表示されます。")
    print("Warning: FEATURE_WEIGHTSは'blobs'データセットのみに適用されています。")
    print("他のデータセットに適用するには、各データセットのn_featuresに合わせてFEATURE_WEIGHTSの次元を調整する必要があります。")