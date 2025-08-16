import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import kl_div

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 常用
plt.rcParams['axes.unicode_minus'] = False

# label
class_ls = [
    '放小球', '擋小球', '殺球', '點扣', '挑球', '防守回挑',
    '長球', '平球', '後場抽平球', '切球', '過渡切球', '推球',
    '撲球', '防守回抽', '勾球', '發短球', '發長球', '未知球種'
]

def label2num(label):
    return class_ls.index(label)

def num2label(num):
    return class_ls[num]

def evaluate_event_level(prediction_df, truth_df, player_mapping, class_ls, save_path=None):
    """
    Event-level evaluation (player + stroke).
    
    Args:
        prediction_df: DataFrame with columns [start_frame, end_frame, stroke, player]
        truth_df: DataFrame with columns [frame, type, player]
        player_mapping: dict mapping prediction player -> truth player, e.g. {0: 'A', 1: 'B'}
        class_ls: list of stroke types
        save_path: path to save results (confusion matrix & report), optional
        
    Returns:
        report (dict), accuracy (float), macro_f1 (float),
        skip_stats (dict: {'skip_zero': ..., 'skip_multi': ...})
    """
    
    truth_labels, pred_labels = [], []
    
    skip_zero = 0   # len == 0
    skip_multi = 0  # len > 1
    
    # 遍歷 ground truth，每個 hit frame 找對應 prediction
    for _, row in truth_df.iterrows():
        target_frame = row['frame_num']
        gt_type = row['type']
        gt_player = row['player']
        
        pred_match = prediction_df[
            (prediction_df['start_frame'] <= target_frame) & 
            (target_frame <= prediction_df['end_frame'])
        ]
        
        if len(pred_match) == 0:
            skip_zero += 1
            continue
        elif len(pred_match) > 1:
            skip_multi += 1
            continue
        
        pred_match = pred_match.iloc[0]
        pred_type = pred_match['stroke'].split('_')[1] if pred_match['stroke']!= '未知球種' else '未知球種'
        pred_player = player_mapping[pred_match['player']]
        
        # event-level label: player + stroke
        gt_event = f"{gt_player}_{gt_type}"
        pred_event = f"{pred_player}_{pred_type}"
        
        truth_labels.append(gt_event)
        pred_labels.append(pred_event)
    
    # 建立 event-level 類別集合
    event_classes = []
    for p in set(truth_df['player']):
        for s in class_ls:
            event_classes.append(f"{p}_{s}")
    
    # classification report
    report = classification_report(
        truth_labels, pred_labels,
        labels=event_classes,
        target_names=event_classes,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    
    accuracy = accuracy_score(truth_labels, pred_labels)        
    macro_f1 = f1_score(truth_labels, pred_labels, average='macro')
    
    # skip 統計
    total_truth = len(truth_df)
    skip_stats = {
        "skip_zero": {"count": skip_zero, "ratio": skip_zero / total_truth},
        "skip_multi": {"count": skip_multi, "ratio": skip_multi / total_truth}
    }
    
    # confusion matrix
    conf = confusion_matrix(truth_labels, pred_labels, labels=event_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf, annot=False, xticklabels=event_classes, yticklabels=event_classes, cmap="Blues")
    plt.title("Event-level Confusion Matrix")
    if save_path:
        plt.savefig(f"{save_path}/events_confusion_matrix.png")
    else:
        plt.show()
    
    # save classification report
    report_df = pd.DataFrame(report).transpose()
    if save_path:
        report_df.to_csv(f"{save_path}/events_classification_report.csv")
    
    return report, accuracy, macro_f1, skip_stats

def evaluate_hit_detection(prediction_df, truth_df, k=1, save_path=None):
    """
    Hit-event level evaluation (忽略球種，只看有無hit)。
    
    Args:
        prediction_df: DataFrame with columns [start_frame, end_frame, stroke, player]
        truth_df: DataFrame with columns [frame, type, player]
        k: 容忍的 hit frames 數量上限
        save_path: optional, path to save classification report
        
    Returns:
        metrics_dict: dict with TP, FP, TN, FN, accuracy, precision, recall, f1
        y_true, y_pred: list of 0/1 for classification report
    """
    
    # 將 truth hit frames存成集合方便檢查
    truth_frames_set = set(truth_df['frame_num'].astype(int).values)
    
    # 先建立每個 frame 的標籤 0/1
    max_frame = max(
        truth_df['frame_num'].max() if not truth_df.empty else 0,
        prediction_df['end_frame'].max() if not prediction_df.empty else 0
    )
    frame_label_true = np.zeros(max_frame+1, dtype=int)
    frame_label_true[list(truth_frames_set)] = 1  # hit=1
    
    # TP, FP, TN, FN 計算
    TP, TN, FP, FN = 0, 0, 0, 0
    
    # 對每個 prediction 片段
    prev_end = 0
    for _, row in prediction_df.iterrows():
        start, end = row['start_frame'], row['end_frame']
        # 計算這段的 truth hit frames
        n_hits = frame_label_true[start:end+1].sum()
        if 1 <= n_hits <= k:
            # TP，將這段 frame 都標成 1
            TP += 1
        else:
            # FP，這段 frame 標成 1，但實際可能沒有或超過 k
            FP += 1

        # TN
        n_hits = frame_label_true[prev_end:start].sum()
        if n_hits == 0:
            TN += 1

        prev_end = end
        frame_label_true[start:end+1] = 0

    # last TN
    n_hits = frame_label_true[prev_end:].sum()
    if n_hits == 0:
        TN += 1
    
    # FN: truth hit frame 被預測片段覆蓋不到
    FN = np.sum(frame_label_true)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP+FP)>0 else 0
    recall = TP / (TP + FN) if (TP+FN)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    # confusion matrix
    conf = [[TP, FP], [FN, TN]]
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf, annot=True, xticklabels=['TP', 'FP'], yticklabels=['FN', 'TN'], cmap="Blues", fmt='d')
    plt.title(f"Hit-event Confusion Matrix (k={k})")
    if save_path:
        plt.savefig(f"{save_path}/hit_events_confusion_matrix.png")
    else:
        plt.show()
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    return metrics_dict

def evaluate_stroke_distribution(prediction_df, truth_df, class_ls, player_mapping, save_path=None, epsilon=1e-8):
    """
    Evaluate stroke distribution with KL divergence.
    
    Args:
        prediction_df: DataFrame with ['stroke', 'player']
        truth_df: DataFrame with ['type', 'player']
        class_ls: list of all stroke types
        player_mapping: dict, e.g. {0: 'A', 1: 'B'} (predict player -> truth player)
        epsilon: small value to avoid log(0) in KL
    
    Returns:
        result_dict: {
            'KL_A': float,
            'KL_B': float,
            'KL_baseline': float,
            'pred_dist_A': pd.Series,
            'pred_dist_B': pd.Series,
            'truth_dist_A': pd.Series,
            'truth_dist_B': pd.Series
        }
    """
    # 統一 player 編號
    prediction_df = prediction_df.copy()
    prediction_df['player'] = prediction_df['player'].map(player_mapping)

    # drop unknown stroke
    prediction_df = prediction_df[prediction_df['stroke']!= '未知球種']
    truth_df = truth_df[truth_df['type']!= '未知球種']
    
    # Ground truth A, B
    truth_df = truth_df.copy()
    truth_A = truth_df[truth_df['player'] == 'A']['type'].value_counts().reindex(class_ls, fill_value=0)
    truth_B = truth_df[truth_df['player'] == 'B']['type'].value_counts().reindex(class_ls, fill_value=0)
    
    # Prediction A, B
    prediction_df = prediction_df.copy()
    prediction_df['stroke'] = prediction_df['stroke'].apply(lambda x: x.split('_')[-1])
    pred_A = prediction_df[prediction_df['player'] == 'A']['stroke'].value_counts().reindex(class_ls, fill_value=0)
    pred_B = prediction_df[prediction_df['player'] == 'B']['stroke'].value_counts().reindex(class_ls, fill_value=0)
    
    # 轉換為機率分布
    truth_A_prob = (truth_A + epsilon) / (truth_A.sum() + epsilon * len(class_ls))
    truth_B_prob = (truth_B + epsilon) / (truth_B.sum() + epsilon * len(class_ls))
    pred_A_prob = (pred_A + epsilon) / (pred_A.sum() + epsilon * len(class_ls))
    pred_B_prob = (pred_B + epsilon) / (pred_B.sum() + epsilon * len(class_ls))

    # Uniform baseline, remove unknown stroke
    uniform_prob = np.ones(len(class_ls)) / (len(class_ls) - 1)
    
    # KL divergence
    KL_A = np.sum(kl_div(truth_A_prob, pred_A_prob))
    KL_B = np.sum(kl_div(truth_B_prob, pred_B_prob))
    KL_A_baseline = np.sum(kl_div(truth_A_prob, uniform_prob))
    KL_B_baseline = np.sum(kl_div(truth_B_prob, uniform_prob))

    # bar plot between KL_A, KL_b, KL_baseline
    plt.figure(figsize=(12, 10))
    labels = ['Player A', 'Player B']
    model_vals = [KL_A, KL_B]
    baseline_vals = [KL_A_baseline, KL_B_baseline]

    x = np.arange(len(labels))  # A, B
    width = 0.35  # 柱寬
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, model_vals, width, label='Model', color='#4C72B0')
    plt.bar(x + width/2, baseline_vals, width, label='Uniform Baseline', color='#C44E52')
    plt.xticks(x, labels)
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence per Player (Model vs Baseline)")
    plt.legend()

    if save_path:
        plt.savefig(f"{save_path}/stroke_distribution_KL.png")
    
    result_dict = {
        'KL_A': KL_A,
        'KL_B': KL_B,
        'KL_A_baseline': KL_A_baseline,
        'KL_B_baseline': KL_B_baseline,
        'pred_dist_A': pred_A_prob,
        'pred_dist_B': pred_B_prob,
        'truth_dist_A': truth_A_prob,
        'truth_dist_B': truth_B_prob
    }
    
    return result_dict

def evaluate(prediction_df, truth_df, player_mapping, class_ls, k=1, save_path=None):
    report, accuracy, macro_f1, skip_stats = evaluate_event_level(prediction_df, truth_df, player_mapping, class_ls, save_path=save_path)
    metrics_dict = evaluate_hit_detection(prediction_df, truth_df, k=k, save_path=save_path)
    result_dict = evaluate_stroke_distribution(prediction_df, truth_df, class_ls, player_mapping, save_path=save_path)

    # Event-level evaluation
    print(f"Event-type evaluation (player + stroke):")
    print(f"\tAccuracy = {accuracy:.4f}, Macro F1 = {macro_f1:.4f}")

    # Skip stats
    print(f"Skip stats:")
    print(f"\t{skip_stats['skip_zero']['count']} has been skipped ({(skip_stats['skip_zero']['ratio'] * 100):.2f}%) since no prediction found.")
    print(f"\t{skip_stats['skip_multi']['count']} has been skipped ({(skip_stats['skip_multi']['ratio'] * 100):.2f}) since multiple events found.")

    # Hit-event evaluation
    print(f"Hit-event evaluation (k={k}):")
    print(f"\tAccuracy = {metrics_dict['accuracy']:.4f}, Precision = {metrics_dict['precision']:.4f}, Recall = {metrics_dict['recall']:.4f}, F1 = {metrics_dict['f1']:.4f}")

    # Stroke distribution evaluation
    print(f"Stroke distribution evaluation:")
    print(f"\tKL_A = {result_dict['KL_A']:.4f}, KL_A_baseline = {result_dict['KL_A_baseline']:.4f}")
    print(f"\tKL_B = {result_dict['KL_B']:.4f}, KL_B_baseline = {result_dict['KL_B_baseline']:.4f}")


name = 'full_1'
prediction_file = f'results/{name}/hit_events.csv'
truth_file = f'ShuttleSet/{name}.csv'
save_path = f'results/{name}'
prediction_df = pd.read_csv(prediction_file)
truth_df = pd.read_csv(truth_file)

# player mapping
# In ShuttleSet, player A is the winner of set
# Adjust this mapping to match your set
player_mapping = {'0': 'A', '1': 'B', 'X': 'X'}

print("Video name:", name)
print("==============================================================================")
evaluate(prediction_df, truth_df, player_mapping, class_ls, k=1, save_path=save_path)
print("==============================================================================")