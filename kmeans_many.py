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

# --- クラスタリングアルゴリズム（正解判定関数利用＆履歴記録つき） ---
def clustering_algorithm_with_correctness(X_stream, k, is_correct_fn, metric='euclidean', initial_X_data_for_kmeans_init=None, weights=None):
    """
    正解判定関数を利用してセントロイドを更新するK-meansのようなクラスタリングアルゴリズム。
    データストリームを順次処理します。
    """
    if initial_X_data_for_kmeans_init is None:
        raise ValueError("initial_X_data_for_kmeans_init must be provided for KMeans++ initialization.")

    C = initialize_centroids(initial_X_data_for_kmeans_init, k)
    N = np.zeros(k) # 各クラスターに割り当てられたデータポイントの数
    centroid_history = [C.copy()]

    for S in X_stream:
        # 各データポイント S を最も近いセントロイドに割り当てる
        dists = [dist(c, S, metric, weights=weights) for c in C]
        min_c = np.argmin(dists) # 割り当てられたクラスターのインデックス

        N[min_c] += 1 # そのクラスターに割り当てられたデータポイント数をカウント

        # 正解判定関数がTrueを返した場合にのみセントロイドを更新
        if is_correct_fn(S, min_c):
            # オンライン学習に似たセントロイド更新（1点ごとの移動平均）
            C[min_c] = C[min_c] + (1 / N[min_c]) * (S - C[min_c])
        
        centroid_history.append(C.copy())
    
    # 最終的なラベル付け (プロット用)
    final_labels = np.zeros(len(initial_X_data_for_kmeans_init), dtype=int)
    for i, S in enumerate(initial_X_data_for_kmeans_init):
        dists = [dist(c, S, metric, weights=weights) for c in C]
        final_labels[i] = np.argmin(dists)

    return C, final_labels, centroid_history

# --- 正解判定関数を生成するファクトリ関数（教師あり） ---
def is_correct_fn_factory(true_centers):
    """
    真のクラスター中心に基づいて、データポイントが正しいクラスターに割り当てられたかを判定する関数を生成します。
    """
    if true_centers is None:
        # 真のクラスター中心がない場合は、常にTrueを返す（ただし、このアルゴリズムでは推奨されない）
        print("Warning: No true_centers provided for correctness check. The algorithm will always consider an assignment 'correct'. This might not be the intended use.")
        return lambda S, assigned_cluster_idx: True

    def is_correct(S, assigned_cluster_idx):
        # データポイント S がどの真のクラスター中心に最も近いかを判断
        true_dists = [np.linalg.norm(tc - S) for tc in true_centers]
        correct_cluster_idx = np.argmin(true_dists)
        
        # アルゴリズムが割り当てたクラスターと真のクラスターが一致するかどうかを返す
        return assigned_cluster_idx == correct_cluster_idx
    return is_correct

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
        X, y_true = wine.data, wine.target 
        k_clusters = len(np.unique(y_true)) # 3クラス
        n_features = X.shape[1] # 13特徴量
    elif dataset_name == 'random':
        # このアルゴリズムでは真のクラスターが存在しないため、適用は推奨されない
        # しかし実験のため生成する
        k_clusters = 3 
        n_features = 2 
        X = np.random.rand(n_samples, n_features) * 10 
        y_true = np.zeros(n_samples) # ダミーのラベル
    elif dataset_name == 'clustered_random':
        k_clusters = 3 
        n_features = 2 
        cluster_std = 1.0 
        
        true_centers = np.array([
            [2, 2],
            [8, 3],
            [5, 8]
        ])
        
        X_list = []
        y_list = []
        for i, center in enumerate(true_centers):
            num_points_in_cluster = n_samples // k_clusters
            X_cluster = np.random.normal(loc=center, scale=cluster_std, size=(num_points_in_cluster, n_features))
            X_list.append(X_cluster)
            y_list.append(np.full(num_points_in_cluster, i)) 

        X = np.vstack(X_list)
        y_true = np.hstack(y_list)
    elif dataset_name == 'overlapping_blobs':
        k_clusters = 3
        n_features = 2 
        n_samples_per_cluster = n_samples // k_clusters
        
        centers = np.array([
            [0, 0],
            [0.5, 0.5],
            [-0.5, 0.5]
        ])
        
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
    
    # 真のセントロイドを計算
    true_centers_calc = None
    if dataset_name == 'random':
        # randomデータには真のクラスターがないため、true_centersはNone
        pass 
    elif dataset_name in ['clustered_random', 'overlapping_blobs']:
        # これらのデータセットはy_trueを基に計算される
        true_centers_calc = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])
    else:
        # その他の標準的なデータセット
        true_centers_calc = np.array([X[y_true == i].mean(axis=0) for i in range(k_clusters)])
    
    return X, y_true, k_clusters, n_features, true_centers_calc

# --- メイン関数 ---
def main(dataset_name: str):
    # データセットの生成
    X, y_true, k_clusters, n_features, true_centers = create_dataset(dataset_name)

    # PCAを適用するかどうかのフラグ (2次元データの場合は不要、高次元の場合に適用)
    apply_pca = n_features > 2

    # ランダムデータセットの場合の警告
    if dataset_name == 'random':
        print(f"Warning: '{dataset_name}' dataset has no inherent clusters. "
              "Applying 'correctness-guided' K-means on this data is generally not meaningful, "
              "as there are no 'true' assignments to guide the algorithm. "
              "The algorithm will try to move centroids towards averaged 'correct' positions based on arbitrary initial assignments.")

    # クラスタリング実行（正解判定関数利用＆セントロイド履歴付き）
    # X_stream として X をイテレータとして渡す
    C_final, final_labels, history = clustering_algorithm_with_correctness(
        X_stream=iter(X),
        k=k_clusters,
        is_correct_fn=is_correct_fn_factory(true_centers), # 真のセントロイドを渡して正解判定関数を生成
        metric='euclidean', # 必要に応じて 'manhattan' や 'cosine' に変更可能
        initial_X_data_for_kmeans_init=X, # K-means++初期化のために全データが必要
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
    # K-meansの教師あり判定なので、真のラベルをプロットして比較できるようにする
    if y_true is not None and dataset_name != 'random': # randomデータはy_trueがダミー
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='True Data (Ground Truth)')
    else:
        # randomデータセットや真のラベルがない場合は、K-meansが割り当てた最終ラベルで色分け
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=final_labels, cmap='viridis', alpha=0.5, label='Clustered Data (K-means Assigned)')
    
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
    ax.set_title(f"Correctness-Guided K-means on {dataset_name.capitalize()} Dataset" + plot_title_suffix)
    plt.legend()
    plt.grid(True)

    def update(frame):
        current_frame = min(frame, len(history_2d) - 1)
        centers_2d = history_2d[current_frame] # 2次元に変換されたセントロイドを取得
        centers_scatter.set_offsets(centers_2d)
        ax.set_title(f"Correctness-Guided K-means on {dataset_name.capitalize()} Dataset - ステップ {current_frame + 1} / {len(history_2d)}" + plot_title_suffix)
        return centers_scatter,

    ani = FuncAnimation(fig, update, frames=len(history_2d), interval=50, blit=True, repeat=False)
    plt.show()

    print(f"--- {dataset_name.capitalize()} Dataset Results (k={k_clusters}) ---")
    print(f"最終的なセントロイド:\n", np.round(C_final, 2))
    if apply_pca:
        print("\n注: プロットはPCAにより2次元に削減されたデータを使用しています。")
    print("-" * 50)


if __name__ == '__main__':
    print("異なるデータセットでの正解判定関数を利用したK-meansクラスタリングを開始します。")

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

    # 真にランダムなデータセット（このアルゴリズムには適しませんが、挙動確認のため）
    main(dataset_name='random')

    print("\n全てのデータセットでの動作確認が完了しました。")
    print("各データセットの結果ウィンドウを閉じると、次のデータセットが表示されます。")
    print("Warning: FEATURE_WEIGHTSは'blobs'データセットのみに適用されています。")
    print("他のデータセットに適用するには、各データセットのn_featuresに合わせてFEATURE_WEIGHTSの次元を調整する必要があります。")