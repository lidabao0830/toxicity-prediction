import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, Set2Set, NNConv
from torch_geometric.utils import add_self_loops
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import copy
import os
from datetime import datetime
import pathlib
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """
    二分类 Focal Loss（基于 logits）
    alpha: 正类权重
    gamma: 聚焦参数
    reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=0.85, gamma=1.7, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)

        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha)
        )

        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def build_loss_function(loss_type='focal', focal_alpha=0.85, focal_gamma=1.7):
    loss_type = str(loss_type).lower().strip()
    if loss_type == 'focal':
        print(f"📌 当前损失函数: Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    elif loss_type == 'bce':
        print("📌 当前损失函数: BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}，仅支持 'focal' 或 'bce'")


class EdgeGCNConv(MessagePassing):
    """边特征增强的图卷积层"""

    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeGCNConv, self).__init__(aggr='add')
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_message = nn.Linear(out_channels * 2, out_channels)
        self.lin_update = nn.Linear(out_channels * 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_node, self.lin_edge, self.lin_message, self.lin_update]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(
            x.size(0), edge_attr.size(1),
            dtype=edge_attr.dtype, device=edge_attr.device
        )
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        x_transformed = self.lin_node(x)
        edge_attr_transformed = self.lin_edge(edge_attr)
        aggr_out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr_transformed)
        out = torch.cat([x_transformed, aggr_out], dim=1)
        out = self.lin_update(F.relu(out))
        return out

    def message(self, x_j, edge_attr):
        fused_features = torch.cat([x_j, edge_attr], dim=1)
        return self.lin_message(fused_features)

    def update(self, aggr_out):
        return aggr_out


class EdgeGCN(nn.Module):
    """边特征增强图卷积神经网络主模型（加入210维分子描述符分支）"""

    def __init__(
            self,
            num_node_features,
            num_edge_features,
            descriptor_dim=210,
            hidden_dim=128,
            dropout=0.3,
            num_layers=2
    ):
        super(EdgeGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.descriptor_dim = descriptor_dim

        self.conv1 = NNConv(
            num_node_features,
            hidden_dim,
            nn=nn.Linear(num_edge_features, num_node_features * hidden_dim)
        )
        self.conv2 = NNConv(
            hidden_dim,
            hidden_dim,
            nn=nn.Linear(num_edge_features, hidden_dim * hidden_dim)
        )
        self.conv3 = NNConv(
            hidden_dim,
            hidden_dim,
            nn=nn.Linear(num_edge_features, hidden_dim * hidden_dim)
        )

        self.use_pool = 'mean'
        if self.use_pool == 'max':
            self.pool = global_max_pool
        elif self.use_pool == 'mean':
            self.pool = global_mean_pool
        elif self.use_pool == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=3)
        else:
            raise ValueError("use_pool should be 'max', 'mean' or 'set2set'")

        pool_out_dim = hidden_dim if self.use_pool in ['max', 'mean'] else 2 * hidden_dim

        self.descriptor_fc1 = nn.Linear(descriptor_dim, 128)
        self.descriptor_fc2 = nn.Linear(128, 64)

        self.fc1 = nn.Linear(pool_out_dim + 2 + 64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def _reshape_graph_feature(self, feat, batch_size, target_dim=None, dtype=None, device=None):
        if feat is None:
            if target_dim is None:
                return None
            return torch.zeros((batch_size, target_dim), dtype=dtype, device=device)

        feat = feat.to(device=device, dtype=dtype)

        if target_dim is None:
            if feat.dim() == 0:
                feat = feat.view(1, 1)
            elif feat.dim() == 1:
                if feat.numel() == batch_size:
                    feat = feat.view(batch_size, 1)
                elif feat.numel() == 1:
                    feat = feat.repeat(batch_size).view(batch_size, 1)
                else:
                    feat = feat.view(-1, 1)
            elif feat.dim() == 2:
                if feat.size(0) == batch_size:
                    pass
                elif feat.size(0) == 1 and batch_size > 1:
                    feat = feat.repeat(batch_size, 1)
                elif feat.numel() == batch_size:
                    feat = feat.view(batch_size, 1)
            return feat

        if feat.dim() == 1:
            if feat.numel() == batch_size * target_dim:
                feat = feat.view(batch_size, target_dim)
            elif feat.numel() == target_dim:
                feat = feat.view(1, target_dim).repeat(batch_size, 1) if batch_size > 1 else feat.view(1, target_dim)
            elif feat.numel() < target_dim:
                pad = torch.zeros(target_dim - feat.numel(), dtype=feat.dtype, device=feat.device)
                feat = torch.cat([feat, pad], dim=0).view(1, target_dim)
                if batch_size > 1:
                    feat = feat.repeat(batch_size, 1)
            else:
                if feat.numel() % target_dim == 0:
                    feat = feat.view(-1, target_dim)
                    if feat.size(0) == batch_size:
                        pass
                    elif feat.size(0) == 1 and batch_size > 1:
                        feat = feat.repeat(batch_size, 1)
                    else:
                        feat = feat[:batch_size]
                else:
                    feat = feat[:target_dim].view(1, target_dim)
                    if batch_size > 1:
                        feat = feat.repeat(batch_size, 1)

        elif feat.dim() == 2:
            if feat.size(0) == batch_size and feat.size(1) == target_dim:
                pass
            elif feat.size(0) == 1 and feat.size(1) == target_dim and batch_size > 1:
                feat = feat.repeat(batch_size, 1)
            elif feat.size(0) == batch_size and feat.size(1) < target_dim:
                pad = torch.zeros((batch_size, target_dim - feat.size(1)), dtype=feat.dtype, device=feat.device)
                feat = torch.cat([feat, pad], dim=1)
            elif feat.size(0) == batch_size and feat.size(1) > target_dim:
                feat = feat[:, :target_dim]
            elif feat.numel() == batch_size * target_dim:
                feat = feat.view(batch_size, target_dim)
            else:
                feat = feat.reshape(-1)
                if feat.numel() >= batch_size * target_dim:
                    feat = feat[:batch_size * target_dim].view(batch_size, target_dim)
                else:
                    padded = torch.zeros(batch_size * target_dim, dtype=feat.dtype, device=feat.device)
                    padded[:feat.numel()] = feat
                    feat = padded.view(batch_size, target_dim)
        else:
            feat = feat.reshape(-1)
            if feat.numel() >= batch_size * target_dim:
                feat = feat[:batch_size * target_dim].view(batch_size, target_dim)
            else:
                padded = torch.zeros(batch_size * target_dim, dtype=feat.dtype, device=feat.device)
                padded[:feat.numel()] = feat
                feat = padded.view(batch_size, target_dim)

        if feat.size(0) != batch_size:
            if feat.size(0) == 1:
                feat = feat.repeat(batch_size, 1)
            else:
                feat = feat[:batch_size]

        if feat.size(1) != target_dim:
            if feat.size(1) < target_dim:
                pad = torch.zeros((feat.size(0), target_dim - feat.size(1)), dtype=feat.dtype, device=feat.device)
                feat = torch.cat([feat, pad], dim=1)
            else:
                feat = feat[:, :target_dim]

        return feat

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = self.dropout(x)

        x = self.pool(x, batch)
        batch_size = x.size(0)

        hba = getattr(data, 'hba', None)
        hbd = getattr(data, 'hbd', None)
        mol_descriptors = getattr(data, 'mol_descriptors', None)

        hba = self._reshape_graph_feature(
            hba, batch_size, target_dim=None, dtype=x.dtype, device=x.device
        )
        hbd = self._reshape_graph_feature(
            hbd, batch_size, target_dim=None, dtype=x.dtype, device=x.device
        )
        mol_descriptors = self._reshape_graph_feature(
            mol_descriptors, batch_size, target_dim=self.descriptor_dim, dtype=x.dtype, device=x.device
        )

        descriptor_feat = F.relu(self.descriptor_fc1(mol_descriptors))
        descriptor_feat = self.dropout(descriptor_feat)
        descriptor_feat = F.relu(self.descriptor_fc2(descriptor_feat))
        descriptor_feat = self.dropout(descriptor_feat)

        x = torch.cat([x, hba, hbd, descriptor_feat], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== 特征提取部分 ====================

def one_hot(val, choices):
    return [float(val == c) for c in choices]


ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRID_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
RING_SIZES = [3, 4, 5, 6, 7, 8]


def atom_features(atom):
    atom_type = one_hot(atom.GetAtomicNum(), ATOM_LIST)
    hyb = one_hot(atom.GetHybridization(), HYBRID_LIST)
    charge = atom.GetFormalCharge() / 3.0
    aromatic = float(atom.GetIsAromatic())
    degree = atom.GetTotalDegree() / 6.0
    explicit_valence = atom.GetExplicitValence() / 6.0
    implicit_valence = atom.GetImplicitValence() / 6.0
    total_h = atom.GetTotalNumHs() / 4.0
    explicit_h = atom.GetNumExplicitHs() / 4.0
    implicit_h = atom.GetNumImplicitHs() / 4.0
    radical_e = atom.GetNumRadicalElectrons() / 2.0
    ring_info = [float(atom.IsInRingSize(rs)) for rs in RING_SIZES]
    in_any_ring = float(atom.IsInRing())
    chirality = one_hot(atom.GetChiralTag(), CHIRALITY_LIST)

    features = (
            atom_type
            + hyb
            + [
                charge, aromatic,
                degree, explicit_valence, implicit_valence,
                total_h, explicit_h, implicit_h,
                radical_e
            ]
            + ring_info
            + [in_any_ring]
            + chirality
    )
    return torch.tensor(features, dtype=torch.float)


BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS
]
BOND_DIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.BEGINWEDGE,
    Chem.rdchem.BondDir.BEGINDASH,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.ENDUPRIGHT
]

_FEAT_DIM_EDGE = len(BOND_TYPES) + 2 + len(STEREO_LIST) + len(BOND_DIR_LIST)


def bond_features(bond):
    btype = one_hot(bond.GetBondType(), BOND_TYPES)
    is_conj = float(bond.GetIsConjugated())
    is_ring = float(bond.IsInRing())
    stereo = one_hot(bond.GetStereo(), STEREO_LIST)
    bdir = one_hot(bond.GetBondDir(), BOND_DIR_LIST)
    features = btype + [is_conj, is_ring] + stereo + bdir
    return torch.tensor(features, dtype=torch.float)


def get_molecular_descriptors(mol):
    """
    提取RDKit分子描述符，并适配为210维
    """
    try:
        if mol is None:
            return None

        descrs = Descriptors.CalcMolDescriptors(mol)
        descriptor_values = np.array(list(descrs.values()), dtype=np.float32)

        descriptor_values = np.where(np.isinf(descriptor_values), 0, descriptor_values)
        descriptor_values = np.where(np.isnan(descriptor_values), 0, descriptor_values)

        target_dim = 210
        if descriptor_values.shape[0] < target_dim:
            descriptor_values = np.pad(
                descriptor_values,
                (0, target_dim - descriptor_values.shape[0]),
                mode='constant',
                constant_values=0
            )
        elif descriptor_values.shape[0] > target_dim:
            descriptor_values = descriptor_values[:target_dim]

        return descriptor_values
    except Exception as e:
        print(f"Error calculating molecular descriptors: {e}")
        return None


def get_atom_features(atom):
    return atom_features(atom).tolist()


def get_bond_features(bond):
    return bond_features(bond).tolist()


# ==================== 性能追踪与可视化模块 ====================

class CrossValidationTracker:
    """交叉验证性能追踪器"""

    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.all_fold_metrics = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [], 'train_rec': [], 'val_rec': [],
            'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': []
        }
        self.fold_best_f1_models = []
        self.fold_best_f1_scores = []
        self.fold_best_f1_epochs = []
        self.fold_best_f1_metrics = []
        self.fold_best_acc_models = []
        self.fold_best_acc_scores = []
        self.fold_best_acc_epochs = []
        self.fold_best_acc_metrics = []

    def add_fold_epoch(self, fold, epoch, train_metrics, val_metrics):
        if len(self.all_fold_metrics['train_loss']) <= fold:
            for key in self.all_fold_metrics:
                self.all_fold_metrics[key].append([])

        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_metrics
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = val_metrics

        metrics = [train_loss, val_loss, train_acc, val_acc, train_prec, val_prec,
                   train_rec, val_rec, train_f1, val_f1, train_auc, val_auc]

        for i, key in enumerate(self.all_fold_metrics.keys()):
            self.all_fold_metrics[key][fold].append(metrics[i])

    def update_fold_best_model(self, fold, epoch, model, val_metrics):
        _, val_acc, val_prec, val_rec, val_f1, val_auc = val_metrics

        while len(self.fold_best_f1_scores) <= fold:
            self.fold_best_f1_scores.append(0.0)
            self.fold_best_f1_models.append(None)
            self.fold_best_f1_epochs.append(0)
            self.fold_best_f1_metrics.append({})

        if val_f1 > self.fold_best_f1_scores[fold]:
            self.fold_best_f1_scores[fold] = val_f1
            self.fold_best_f1_models[fold] = copy.deepcopy(model.state_dict())
            self.fold_best_f1_epochs[fold] = epoch
            self.fold_best_f1_metrics[fold] = {
                'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec,
                'f1': val_f1, 'auc': val_auc
            }

        while len(self.fold_best_acc_scores) <= fold:
            self.fold_best_acc_scores.append(0.0)
            self.fold_best_acc_models.append(None)
            self.fold_best_acc_epochs.append(0)
            self.fold_best_acc_metrics.append({})

        if val_acc > self.fold_best_acc_scores[fold]:
            self.fold_best_acc_scores[fold] = val_acc
            self.fold_best_acc_models[fold] = copy.deepcopy(model.state_dict())
            self.fold_best_acc_epochs[fold] = epoch
            self.fold_best_acc_metrics[fold] = {
                'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec,
                'f1': val_f1, 'auc': val_auc
            }

    def print_fold_summary(self, fold):
        print(f"\n{'=' * 60}")
        print(f"第 {fold + 1} 折交叉验证结果总结")
        print(f"{'=' * 60}")

        best_f1_metrics = self.fold_best_f1_metrics[fold]
        best_f1_epoch = self.fold_best_f1_epochs[fold]
        print(f"🏆 最佳F1分数模型 (Epoch {best_f1_epoch + 1}):")
        for key, value in best_f1_metrics.items():
            print(f"   {key.capitalize()}: {value:.4f}")

        best_acc_metrics = self.fold_best_acc_metrics[fold]
        best_acc_epoch = self.fold_best_acc_epochs[fold]
        print(f"\n⭐ 最佳准确率模型 (Epoch {best_acc_epoch + 1}):")
        for key, value in best_acc_metrics.items():
            print(f"   {key.capitalize()}: {value:.4f}")

    def print_overall_summary(self):
        print(f"\n{'=' * 80}")
        print(f"{self.n_folds}折交叉验证总体结果")
        print(f"{'=' * 80}")

        metric_names = ['f1', 'accuracy', 'precision', 'recall', 'auc']
        f1_metrics_data = {name: [m[name] for m in self.fold_best_f1_metrics] for name in metric_names}
        print(f"\n📊 各折最佳F1分数模型的性能统计:")
        for name, values in f1_metrics_data.items():
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"   {name.capitalize()} - 均值: {mean_val:.4f} ± {std_val:.4f}")

        acc_metrics_data = {name: [m[name] for m in self.fold_best_acc_metrics] for name in metric_names}
        print(f"\n📊 各折最佳准确率模型的性能统计:")
        for name, values in acc_metrics_data.items():
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"   {name.capitalize()} - 均值: {mean_val:.4f} ± {std_val:.4f}")

        print(f"\n📈 各折最佳F1模型详细结果:")
        for fold in range(self.n_folds):
            metrics = self.fold_best_f1_metrics[fold]
            epoch = self.fold_best_f1_epochs[fold]
            print(f"   第{fold + 1}折 (Epoch {epoch + 1}): F1={metrics['f1']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}, AUC={metrics['auc']:.4f}")

        print(f"\n📈 各折最佳准确率模型详细结果:")
        for fold in range(self.n_folds):
            metrics = self.fold_best_acc_metrics[fold]
            epoch = self.fold_best_acc_epochs[fold]
            print(f"   第{fold + 1}折 (Epoch {epoch + 1}): Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}, AUC={metrics['auc']:.4f}")

    def plot_cv_results(self):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle(f'{self.n_folds}折交叉验证结果', fontsize=16, y=0.98)

        base_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = [base_colors[i % len(base_colors)] for i in range(self.n_folds)]

        # 为普通训练(n_folds=1)定义专用颜色
        train_color = '#2E86DE'  # 蓝色
        val_color = '#EE5A6F'    # 红色

        ax = axes[0, 0]
        for fold in range(self.n_folds):
            epochs = range(1, len(self.all_fold_metrics['train_loss'][fold]) + 1)

            if self.n_folds == 1:
                # 普通训练模式:训练用蓝色实线,验证用红色实线
                ax.plot(epochs, self.all_fold_metrics['train_loss'][fold],
                        color=train_color, linewidth=2.5, linestyle='-',
                        label='训练损失', alpha=0.9)
                ax.plot(epochs, self.all_fold_metrics['val_loss'][fold],
                        color=val_color, linewidth=2.5, linestyle='-',
                        label='验证损失', alpha=0.9)
            else:
                # 交叉验证模式:保持原样
                ax.plot(epochs, self.all_fold_metrics['train_loss'][fold],
                        color=colors[fold], alpha=0.7, linestyle='-',
                        label=f'Fold {fold + 1} Train')
                ax.plot(epochs, self.all_fold_metrics['val_loss'][fold],
                        color=colors[fold], alpha=0.7, linestyle='--',
                        label=f'Fold {fold + 1} Val')

        ax.set_title('训练和验证损失')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        metric_key_mapping = {
            'val_acc': 'accuracy',
            'val_f1': 'f1',
            'val_auc': 'auc',
            'val_prec': 'precision',
            'val_rec': 'recall'
        }

        metrics_to_plot = [
            ('val_acc', '验证准确率', 'Accuracy'),
            ('val_f1', '验证F1分数', 'F1 Score'),
            ('val_auc', '验证AUC-ROC', 'AUC-ROC')
        ]
        plot_positions = [(0, 1), (1, 0), (1, 1)]

        for (metric_key, title, ylabel), (row, col) in zip(metrics_to_plot, plot_positions):
            ax = axes[row, col]
            for fold in range(self.n_folds):
                epochs = range(1, len(self.all_fold_metrics[metric_key][fold]) + 1)

                if self.n_folds == 1:
                    # 普通训练模式:用红色实线
                    ax.plot(epochs, self.all_fold_metrics[metric_key][fold],
                            color=val_color, linewidth=2.5, label='验证集')
                else:
                    # 交叉验证模式:保持原样
                    ax.plot(epochs, self.all_fold_metrics[metric_key][fold],
                            color=colors[fold], linewidth=2, label=f'Fold {fold + 1}')

                best_epoch = self.fold_best_f1_epochs[fold]
                metric_name = metric_key_mapping[metric_key]
                best_val = self.fold_best_f1_metrics[fold][metric_name]

                if self.n_folds == 1:
                    # 普通训练模式:最佳点用金色星标
                    ax.scatter(best_epoch + 1, best_val, color='#F39C12',
                               s=150, marker='*', edgecolor='black', linewidth=1.5,
                               label='最佳点', zorder=5)
                else:
                    # 交叉验证模式:保持原样
                    ax.scatter(best_epoch + 1, best_val, color=colors[fold],
                               s=100, marker='*', edgecolor='black', linewidth=1)

            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        for fold in range(self.n_folds):
            epochs = range(1, len(self.all_fold_metrics['val_prec'][fold]) + 1)

            if self.n_folds == 1:
                # 普通训练模式:精确率用绿色,召回率用橙色
                prec_color = '#10AC84'  # 绿色
                rec_color = '#F39C12'   # 橙色
                ax.plot(epochs, self.all_fold_metrics['val_prec'][fold],
                        color=prec_color, linewidth=2.5, linestyle='-',
                        label='验证精确率')
                ax.plot(epochs, self.all_fold_metrics['val_rec'][fold],
                        color=rec_color, linewidth=2.5, linestyle='-',
                        label='验证召回率')
            else:
                # 交叉验证模式:保持原样
                ax.plot(epochs, self.all_fold_metrics['val_prec'][fold],
                        color=colors[fold], linewidth=2, linestyle='-',
                        label=f'Fold {fold + 1} Prec')
                ax.plot(epochs, self.all_fold_metrics['val_rec'][fold],
                        color=colors[fold], linewidth=2, linestyle='--',
                        label=f'Fold {fold + 1} Rec')

        ax.set_title('验证精确率和召回率')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        max_epochs = max(len(fold_metrics) for fold_metrics in self.all_fold_metrics['val_acc'])
        avg_metrics = {'val_acc': [], 'val_f1': [], 'val_auc': []}
        for epoch in range(max_epochs):
            for metric in avg_metrics:
                epoch_values = []
                for fold in range(self.n_folds):
                    if epoch < len(self.all_fold_metrics[metric][fold]):
                        epoch_values.append(self.all_fold_metrics[metric][fold][epoch])
                avg_metrics[metric].append(np.mean(epoch_values) if epoch_values else 0)

        epochs = range(1, len(avg_metrics['val_acc']) + 1)

        if self.n_folds == 1:
            # 普通训练模式:显示训练集和验证集的对比
            ax.plot(epochs, self.all_fold_metrics['train_acc'][0],
                    color=train_color, linewidth=2.5, label='训练准确率', linestyle='-')
            ax.plot(epochs, avg_metrics['val_acc'],
                    color=val_color, linewidth=2.5, label='验证准确率', linestyle='-')
            ax.plot(epochs, self.all_fold_metrics['train_f1'][0],
                    color='#10AC84', linewidth=2.5, label='训练F1分数', linestyle='-')
            ax.plot(epochs, avg_metrics['val_f1'],
                    color='#F39C12', linewidth=2.5, label='验证F1分数', linestyle='-')
            ax.set_title('训练集与验证集性能对比')
        else:
            # 交叉验证模式:保持原样
            ax.plot(epochs, avg_metrics['val_acc'], 'b-', linewidth=3, label='平均准确率')
            ax.plot(epochs, avg_metrics['val_f1'], 'r-', linewidth=3, label='平均F1分数')
            ax.plot(epochs, avg_metrics['val_auc'], 'g-', linewidth=3, label='平均AUC')
            ax.set_title('各折平均性能指标')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def smiles_to_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def mol_to_graph(mol, label=None):
    if mol is None:
        return None

    mol_descriptors = get_molecular_descriptors(mol)
    if mol_descriptors is None:
        return None

    atom_feature_list = [atom_features(atom) for atom in mol.GetAtoms()]
    if len(atom_feature_list) == 0:
        return None
    x = torch.stack(atom_feature_list, dim=0)

    edge_indices = []
    edge_feature_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_feature_list.extend([bond_feat, bond_feat])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_feature_list, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, _FEAT_DIM_EDGE), dtype=torch.float)

    mol_hba = float(rdMolDescriptors.CalcNumHBA(mol))
    mol_hbd = float(rdMolDescriptors.CalcNumHBD(mol))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        hba=torch.tensor([mol_hba], dtype=torch.float),
        hbd=torch.tensor([mol_hbd], dtype=torch.float),
        mol_descriptors=torch.tensor(mol_descriptors, dtype=torch.float)
    )

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


class FeatureNormalizer:
    """特征标准化器"""

    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.descriptor_scaler = StandardScaler()
        self.fitted = False

    def fit(self, data_list):
        all_node_features = []
        all_edge_features = []
        all_descriptors = []

        for data in data_list:
            if data is not None:
                all_node_features.append(data.x.numpy())
                if data.edge_attr.size(0) > 0:
                    all_edge_features.append(data.edge_attr.numpy())

                if hasattr(data, 'mol_descriptors'):
                    all_descriptors.append(data.mol_descriptors.numpy().reshape(1, -1))

        if all_node_features:
            self.node_scaler.fit(np.vstack(all_node_features))
        if all_edge_features:
            self.edge_scaler.fit(np.vstack(all_edge_features))
        if all_descriptors:
            self.descriptor_scaler.fit(np.vstack(all_descriptors))

        self.fitted = True

    def transform(self, data_list):
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        normalized_data_list = []
        for data in data_list:
            if data is None:
                continue

            normalized_data = data.clone()
            normalized_data.x = torch.tensor(
                self.node_scaler.transform(data.x.numpy()), dtype=torch.float
            )

            if data.edge_attr.size(0) > 0:
                normalized_data.edge_attr = torch.tensor(
                    self.edge_scaler.transform(data.edge_attr.numpy()), dtype=torch.float
                )

            if hasattr(data, 'mol_descriptors'):
                normalized_data.mol_descriptors = torch.tensor(
                    self.descriptor_scaler.transform(data.mol_descriptors.numpy().reshape(1, -1)).flatten(),
                    dtype=torch.float
                )

            normalized_data_list.append(normalized_data)
        return normalized_data_list

    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)


def train_model(model, train_loader, criterion, optimizer, device):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_model(model, loader, criterion, device):
    """评估模型性能"""
    model.eval()
    predictions, true_labels = [], []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1))
            total_loss += loss.item()
            num_batches += 1
            pred = torch.sigmoid(out).view(-1).cpu().numpy()
            predictions.extend(pred.flatten())
            true_labels.extend(batch.y.view(-1).cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    binary_pred = (predictions > 0.5).astype(int)

    metrics = [
        avg_loss,
        accuracy_score(true_labels, binary_pred),
        precision_score(true_labels, binary_pred, zero_division=0),
        recall_score(true_labels, binary_pred, zero_division=0),
        f1_score(true_labels, binary_pred, zero_division=0),
        roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0.5
    ]
    return metrics


def run_cross_validation(graphs, n_folds=5, n_epochs=200, batch_size=64,
                         hidden_dim=128, dropout=0.3, learning_rate=0.0001,
                         weight_decay=1e-4, num_layers=2, device='cuda',
                         loss_type='focal', focal_alpha=0.85, focal_gamma=1.7,
                         progress_callback=None):
    """执行交叉验证训练（支持UI进度回调）"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_tracker = CrossValidationTracker(n_folds)
    labels = [g.y.item() for g in graphs]
    indices = np.arange(len(graphs))

    print(f"开始 {n_folds} 折交叉验证...")
    print(f"总样本数: {len(graphs)}")
    print(f"正样本: {labels.count(1)}, 负样本: {labels.count(0)}")

    if progress_callback:
        progress_callback(stage='start_cv', n_folds=n_folds, n_epochs=n_epochs)

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if progress_callback:
            progress_callback(stage='fold_start', fold=fold, n_folds=n_folds, n_epochs=n_epochs)

        print(f"\n{'=' * 60}")
        print(f"第 {fold + 1} 折交叉验证开始")
        print(f"{'=' * 60}")

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_labels = [g.y.item() for g in train_graphs]
        val_labels = [g.y.item() for g in val_graphs]

        print(f"训练集: 总数={len(train_graphs)}, 正样本={train_labels.count(1)}, 负样本={train_labels.count(0)}")
        print(f"验证集: 总数={len(val_graphs)}, 正样本={val_labels.count(1)}, 负样本={val_labels.count(0)}")

        fold_normalizer = FeatureNormalizer()
        train_graphs_norm = fold_normalizer.fit_transform(train_graphs)
        val_graphs_norm = fold_normalizer.transform(val_graphs)

        train_loader = DataLoader(train_graphs_norm, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs_norm, batch_size=batch_size)

        num_node_features = graphs[0].x.size(1)
        num_edge_features = graphs[0].edge_attr.size(1) if graphs[0].edge_attr.size(0) > 0 else _FEAT_DIM_EDGE
        descriptor_dim = graphs[0].mol_descriptors.size(0) if hasattr(graphs[0], 'mol_descriptors') else 210

        model = EdgeGCN(
            num_node_features,
            num_edge_features,
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = build_loss_function(loss_type, focal_alpha, focal_gamma)

        print(f"\n开始训练第 {fold + 1} 折模型...")

        for epoch in range(n_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            train_metrics = evaluate_model(model, train_loader, criterion, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)

            cv_tracker.add_fold_epoch(fold, epoch, train_metrics, val_metrics)
            cv_tracker.update_fold_best_model(fold, epoch, model, val_metrics)

            if progress_callback:
                progress_callback(
                    stage='epoch_end',
                    fold=fold,
                    epoch=epoch,
                    n_folds=n_folds,
                    n_epochs=n_epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )

            if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
                print(f'  Epoch {epoch + 1:03d}/{n_epochs}:')
                print(
                    f'    Train - Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.4f}, Prec: {train_metrics[2]:.4f}, Rec: {train_metrics[3]:.4f}, F1: {train_metrics[4]:.4f}, AUC: {train_metrics[5]:.4f}')
                print(
                    f'    Val   - Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.4f}, Prec: {val_metrics[2]:.4f}, Rec: {val_metrics[3]:.4f}, F1: {val_metrics[4]:.4f}, AUC: {val_metrics[5]:.4f}')
                if cv_tracker.fold_best_f1_scores[fold] > 0:
                    best_f1 = cv_tracker.fold_best_f1_scores[fold]
                    best_f1_epoch = cv_tracker.fold_best_f1_epochs[fold]
                    best_acc = cv_tracker.fold_best_acc_scores[fold]
                    best_acc_epoch = cv_tracker.fold_best_acc_epochs[fold]
                    print(f'    当前最佳F1: {best_f1:.4f} (Epoch {best_f1_epoch + 1})')
                    print(f'    当前最佳Acc: {best_acc:.4f} (Epoch {best_acc_epoch + 1})')
                print()

        if progress_callback:
            progress_callback(stage='fold_end', fold=fold, n_folds=n_folds, n_epochs=n_epochs)

        cv_tracker.print_fold_summary(fold)

    cv_tracker.print_overall_summary()
    cv_tracker.plot_cv_results()

    if progress_callback:
        progress_callback(stage='end_cv', n_folds=n_folds, n_epochs=n_epochs)

    return cv_tracker


def run_standard_training(graphs, test_size=0.2, random_state=42,
                          n_epochs=200, batch_size=64, hidden_dim=128,
                          dropout=0.3, learning_rate=0.0001, weight_decay=1e-4,
                          num_layers=2, device='cuda',
                          loss_type='focal', focal_alpha=0.85, focal_gamma=1.7,
                          progress_callback=None):
    """执行普通训练评估（训练集/验证集拆分，支持UI进度回调）"""
    from sklearn.model_selection import train_test_split

    train_graphs, val_graphs = train_test_split(
        graphs,
        test_size=test_size,
        random_state=random_state,
        stratify=[g.y.item() for g in graphs]
    )

    cv_tracker = CrossValidationTracker(n_folds=1)

    train_labels = [g.y.item() for g in train_graphs]
    val_labels = [g.y.item() for g in val_graphs]

    print(f"\n{'=' * 60}")
    print(f"开始普通训练评估（训练集+验证集）")
    print(f"{'=' * 60}")
    print(f"训练集: 总数={len(train_graphs)}, 正样本={train_labels.count(1)}, 负样本={train_labels.count(0)}")
    print(f"验证集: 总数={len(val_graphs)}, 正样本={val_labels.count(1)}, 负样本={val_labels.count(0)}")

    if progress_callback:
        progress_callback(stage='start_standard', n_folds=1, n_epochs=n_epochs)
        progress_callback(stage='fold_start', fold=0, n_folds=1, n_epochs=n_epochs)

    fold_normalizer = FeatureNormalizer()
    train_graphs_norm = fold_normalizer.fit_transform(train_graphs)
    val_graphs_norm = fold_normalizer.transform(val_graphs)

    train_loader = DataLoader(train_graphs_norm, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs_norm, batch_size=batch_size)

    num_node_features = graphs[0].x.size(1)
    num_edge_features = graphs[0].edge_attr.size(1) if graphs[0].edge_attr.size(0) > 0 else _FEAT_DIM_EDGE
    descriptor_dim = graphs[0].mol_descriptors.size(0) if hasattr(graphs[0], 'mol_descriptors') else 210

    model = EdgeGCN(
        num_node_features,
        num_edge_features,
        descriptor_dim=descriptor_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = build_loss_function(loss_type, focal_alpha, focal_gamma)

    print(f"\n开始训练模型...")

    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate_model(model, train_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        cv_tracker.add_fold_epoch(0, epoch, train_metrics, val_metrics)
        cv_tracker.update_fold_best_model(0, epoch, model, val_metrics)

        if progress_callback:
            progress_callback(
                stage='epoch_end',
                fold=0,
                epoch=epoch,
                n_folds=1,
                n_epochs=n_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )

        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            print(f'  Epoch {epoch + 1:03d}/{n_epochs}:')
            print(
                f'    Train - Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.4f}, Prec: {train_metrics[2]:.4f}, Rec: {train_metrics[3]:.4f}, F1: {train_metrics[4]:.4f}, AUC: {train_metrics[5]:.4f}')
            print(
                f'    Val   - Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.4f}, Prec: {val_metrics[2]:.4f}, Rec: {val_metrics[3]:.4f}, F1: {val_metrics[4]:.4f}, AUC: {val_metrics[5]:.4f}')
            if cv_tracker.fold_best_f1_scores[0] > 0:
                best_f1 = cv_tracker.fold_best_f1_scores[0]
                best_f1_epoch = cv_tracker.fold_best_f1_epochs[0]
                best_acc = cv_tracker.fold_best_acc_scores[0]
                best_acc_epoch = cv_tracker.fold_best_acc_epochs[0]
                print(f'    当前最佳F1: {best_f1:.4f} (Epoch {best_f1_epoch + 1})')
                print(f'    当前最佳Acc: {best_acc:.4f} (Epoch {best_acc_epoch + 1})')
            print()

    if progress_callback:
        progress_callback(stage='fold_end', fold=0, n_folds=1, n_epochs=n_epochs)
        progress_callback(stage='end_standard', n_folds=1, n_epochs=n_epochs)

    cv_tracker.print_fold_summary(0)
    cv_tracker.print_overall_summary()
    cv_tracker.plot_cv_results()

    return cv_tracker


class ToxicityPredictionPlatform:
    """化合物毒性预测系统主类"""

    def __init__(self, model_save_dir='./trained_models'):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.model = None
        self.normalizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_node_features = None
        self.num_edge_features = None
        self.descriptor_dim = 210
        self.training_config = {}

        print(f"✅ 化合物毒性预测系统已初始化")
        print(f"📁 模型保存目录: {model_save_dir}")
        print(f"💻 使用设备: {self.device}")
    def _normalize_model_save_dir(self, save_dir):

        save_dir = str(save_dir).strip().strip('"').strip("'")

        if not save_dir:
            save_dir = "./trained_models"

        # Windows 下若用户输入 D:，自动修正为 D:\
        if os.name == "nt" and len(save_dir) == 2 and save_dir[1] == ":":
            save_dir = save_dir + "\\"

        return os.path.normpath(save_dir)
    def load_and_preprocess_data(self, csv_file, smiles_col='SMILES', label_col='Toxicity_Label'):
        """加载并预处理数据"""
        print(f"\n{'=' * 60}")
        print(f"📂 正在加载数据: {csv_file}")
        print(f"{'=' * 60}")

        data = pd.read_csv(csv_file)
        data = data.dropna(subset=[smiles_col])
        print(f"原始数据条数: {len(data)}")

        data[smiles_col] = data[smiles_col].apply(canonicalize_smiles)
        data = data.dropna(subset=[smiles_col])

        data['mol'] = data[smiles_col].apply(smiles_to_mol)
        valid_mols = data['mol'].notna().sum()
        print(f"有效分子数: {valid_mols} / {len(data)}")

        unique_labels = data[label_col].unique()
        print(f"检测到的标签类别: {unique_labels}")
        if set(unique_labels).issubset({0, 1}):
            print("标签已为二进制格式 (0/1)")
        else:
            toxicity_map = {
                'Non-toxic': 0, 'Toxicity': 1, 'non-toxic': 0, 'toxic': 1,
                'negative': 0, 'positive': 1, 'False': 0, 'True': 1
            }
            data[label_col] = data[label_col].map(toxicity_map)
            print("标签已映射为二进制格式")

        graphs = []
        for _, row in data.iterrows():
            if row['mol'] is not None:
                graph = mol_to_graph(row['mol'], row[label_col])
                if graph is not None:
                    graphs.append(graph)

        labels = [g.y.item() for g in graphs]
        print(f"\n✅ 数据预处理完成")
        print(f"总图数据: {len(graphs)}")
        print(f"有毒样本: {labels.count(1)}, 无毒样本: {labels.count(0)}")
        if len(graphs) > 0 and hasattr(graphs[0], 'mol_descriptors'):
            print(f"分子描述符维度: {graphs[0].mol_descriptors.size(0)}")
        return graphs

    def train(self, graphs, training_mode='cv', n_folds=5, test_size=0.2, random_state=42,
              n_epochs=200, batch_size=64, hidden_dim=128, dropout=0.3,
              learning_rate=0.0001, weight_decay=1e-4, num_layers=2,
              loss_type='focal', focal_alpha=0.85, focal_gamma=1.7,
              progress_callback=None):
        print(f"\n{'=' * 80}")
        print(f"🚀 开始模型训练（模式：{training_mode}）")
        print(f"{'=' * 80}")

        self.training_config = {
            'training_mode': training_mode,
            'n_folds': n_folds,
            'test_size': test_size,
            'random_state': random_state,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_layers': num_layers,
            'loss_type': loss_type,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma
        }

        print("📋 本次训练配置:")
        for key, value in self.training_config.items():
            print(f"   {key}: {value}")

        if training_mode == 'cv':
            cv_tracker = run_cross_validation(
                graphs=graphs,
                n_folds=n_folds,
                n_epochs=n_epochs,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_layers=num_layers,
                device=self.device,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                progress_callback=progress_callback
            )
        elif training_mode == 'standard':
            cv_tracker = run_standard_training(
                graphs=graphs,
                test_size=test_size,
                random_state=random_state,
                n_epochs=n_epochs,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_layers=num_layers,
                device=self.device,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                progress_callback=progress_callback
            )
        else:
            raise ValueError(f"不支持的训练模式：{training_mode}，仅支持 'cv' 或 'standard'")

        best_fold = np.argmax(cv_tracker.fold_best_acc_scores)
        print(f"\n🏆 选择第 {best_fold + 1} 折作为最终预测模型")
        print(f"   选择依据: 最高准确率 = {cv_tracker.fold_best_acc_scores[best_fold]:.4f}")
        print(f"   该模型性能指标:")
        for key, value in cv_tracker.fold_best_acc_metrics[best_fold].items():
            print(f"      {key.capitalize()}: {value:.4f}")

        self.num_node_features = graphs[0].x.size(1)
        self.num_edge_features = graphs[0].edge_attr.size(1) if graphs[0].edge_attr.size(0) > 0 else _FEAT_DIM_EDGE
        self.descriptor_dim = graphs[0].mol_descriptors.size(0) if hasattr(graphs[0], 'mol_descriptors') else 210

        self.model = EdgeGCN(
            self.num_node_features,
            self.num_edge_features,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers
        ).to(self.device)

        self.model.load_state_dict(cv_tracker.fold_best_acc_models[best_fold])

        self.normalizer = FeatureNormalizer()
        self.normalizer.fit(graphs)

        self.model.eval()

        print(f"✅ 模型训练完成!")
        return cv_tracker

    def save_model(self, model_name=None):
        """保存训练好的模型"""
        if self.model is None or self.normalizer is None:
            raise ValueError("没有训练好的模型可保存，请先完成训练")

        # 处理模型名称
        if model_name is None or str(model_name).strip() == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"toxicity_model_{timestamp}"
        else:
            model_name = str(model_name).strip()

        # 去掉用户可能手动输入的 .pth 后缀
        model_name = pathlib.Path(model_name).stem

        # 去掉 Windows 文件名非法字符
        invalid_chars = r'\/:*?"<>|'
        model_name = "".join("_" if c in invalid_chars else c for c in model_name).strip()

        if not model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"toxicity_model_{timestamp}"

        # 规范化并确保保存目录存在
        self.model_save_dir = self._normalize_model_save_dir(self.model_save_dir)
        pathlib.Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)

        save_path = os.path.join(self.model_save_dir, f"{model_name}.pth")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'normalizer': self.normalizer,
            'num_node_features': self.num_node_features,
            'num_edge_features': self.num_edge_features,
            'descriptor_dim': self.descriptor_dim,
            'training_config': self.training_config,
            'device': str(self.device)
        }

        torch.save(checkpoint, save_path)
        print(f"✅ 模型已保存至: {save_path}")
        return save_path

    def load_model(self, model_path):
        """加载已保存的模型"""
        print(f"📥 正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.num_node_features = checkpoint['num_node_features']
        self.num_edge_features = checkpoint['num_edge_features']
        self.descriptor_dim = checkpoint.get('descriptor_dim', 210)
        self.normalizer = checkpoint['normalizer']
        self.training_config = checkpoint.get('training_config', {})

        hidden_dim = self.training_config.get('hidden_dim', 128)
        dropout = self.training_config.get('dropout', 0.3)
        num_layers = self.training_config.get('num_layers', 2)

        self.model = EdgeGCN(
            self.num_node_features,
            self.num_edge_features,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ 模型加载成功!")
        if self.training_config:
            print(f"📋 训练配置信息:")
            for key, value in self.training_config.items():
                print(f"   {key}: {value}")

    def predict_single(self, smiles_string):
        """预测单个化合物的毒性"""
        if self.model is None or self.normalizer is None:
            print("❌ 错误: 请先训练或加载模型")
            return None

        canonical_smiles = canonicalize_smiles(smiles_string)
        if canonical_smiles is None:
            print(f"❌ 无效的SMILES: {smiles_string}")
            return None

        mol = smiles_to_mol(canonical_smiles)
        if mol is None:
            print(f"❌ 无法解析分子: {smiles_string}")
            return None

        graph = mol_to_graph(mol)
        if graph is None:
            print(f"❌ 无法转换为图数据: {smiles_string}")
            return None

        normalized_graph = self.normalizer.transform([graph])[0]

        self.model.eval()
        with torch.no_grad():
            normalized_graph = normalized_graph.to(self.device)
            normalized_graph.batch = torch.zeros(
                normalized_graph.x.size(0),
                dtype=torch.long,
                device=self.device
            )
            output = self.model(normalized_graph)
            probability = torch.sigmoid(output).item()
            prediction = "有毒" if probability > 0.5 else "无毒"

        return {
            'smiles': canonical_smiles,
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability, 1 - probability)
        }

    def predict_batch(self, smiles_list):
        """批量预测多个化合物"""
        results = []
        print(f"\n{'=' * 60}")
        print(f"🔬 开始批量预测 (共{len(smiles_list)}个化合物)")
        print(f"{'=' * 60}\n")
        for i, smiles in enumerate(smiles_list, 1):
            result = self.predict_single(smiles)
            if result:
                results.append(result)
                print(f"{i}. {result['smiles']}")
                print(
                    f"   预测结果: {result['prediction']} (概率: {result['probability']:.4f}, 置信度: {result['confidence']:.4f})\n")
        return results

    def predict_batch_from_csv(self, input_csv_path, output_csv_path, smiles_col='SMILES'):
        """
        从CSV文件批量预测化合物毒性并保存结果
        """
        if self.model is None or self.normalizer is None:
            print("❌ 错误: 请先训练或加载模型")
            return None

        print(f"\n{'=' * 80}")
        print(f"📂 正在从CSV文件加载数据: {input_csv_path}")
        print(f"{'=' * 80}")

        try:
            data = pd.read_csv(input_csv_path)
        except Exception as e:
            print(f"❌ 读取CSV文件失败: {e}")
            return None

        if smiles_col not in data.columns:
            print(f"❌ 错误: 列 '{smiles_col}' 不存在于数据中")
            print(f"可用列名: {list(data.columns)}")
            return None

        print(f"✅ 成功加载数据,共 {len(data)} 条记录")
        print(f"📊 开始批量预测...")

        predictions = []
        probabilities = []
        confidences = []

        success_count = 0
        fail_count = 0

        for idx, row in data.iterrows():
            smiles = row[smiles_col]

            if pd.isna(smiles) or smiles == '':
                predictions.append('无效')
                probabilities.append(None)
                confidences.append(None)
                fail_count += 1
                continue

            result = self.predict_single(smiles)
            if result:
                predictions.append(result['prediction'])
                probabilities.append(result['probability'])
                confidences.append(result['confidence'])
                success_count += 1
            else:
                predictions.append('无效')
                probabilities.append(None)
                confidences.append(None)
                fail_count += 1

            if (idx + 1) % 100 == 0:
                print(f"   已处理: {idx + 1}/{len(data)} 条")

        data['Toxicity_Prediction'] = predictions
        data['Toxicity_Probability'] = probabilities
        data['Prediction_Confidence'] = confidences

        output_csv_path = output_csv_path.strip().strip('"').strip("'")
        output_path = pathlib.Path(output_csv_path)

        if output_path.exists() and output_path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_results_{timestamp}.csv"
            final_output_path = output_path / filename
            print(f"\n📝 检测到目录路径, 将在该目录下生成文件: {filename}")

        elif not output_path.exists():
            if output_path.suffix.lower() == '.csv':
                final_output_path = output_path
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"\n📝 检测到文件路径, 将创建目录并保存: {final_output_path}")
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prediction_results_{timestamp}.csv"
                final_output_path = output_path / filename
                print(f"\n📝 检测到目录路径(不存在), 创建目录并生成文件: {filename}")

        elif output_path.exists() and output_path.is_file():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{output_path.stem}_{timestamp}{output_path.suffix}"
            final_output_path = output_path.parent / new_filename
            print(f"\n📝 输出文件已存在，自动保存为新文件: {new_filename}")

        final_output_str = str(final_output_path.absolute())

        try:
            data.to_csv(
                final_output_str,
                index=False,
                encoding='utf-8-sig'
            )

            print(f"\n{'=' * 80}")
            print(f"✅ 批量预测完成!")
            print(f"{'=' * 80}")
            print(f"📊 预测统计:")
            print(f"   - 总记录数: {len(data)}")
            print(f"   - 成功预测: {success_count}")
            print(f"   - 预测失败: {fail_count}")
            print(f"\n💾 结果已保存至: {final_output_str}")
            print(f"\n📋 输出列说明:")
            print(f"   - Toxicity_Prediction: 预测结果(有毒/无毒/无效)")
            print(f"   - Toxicity_Probability: 有毒概率 (0-1)")
            print(f"   - Prediction_Confidence: 预测置信度 (0.5-1)")
            print(f"{'=' * 80}\n")
            return data

        except Exception as e:
            print(f"\n❌ 保存结果失败: {e}")
            print(f"📌 保存路径: {final_output_str}")
            print(f"💡 可能的解决方案:")
            print(f"   1. 检查路径是否有权限写入")
            print(f"   2. 确保磁盘有足够空间")
            print(f"   3. 尝试使用绝对路径而非相对路径")
            return None

    def interactive_prediction(self):
        """交互式预测界面"""
        print(f"\n{'=' * 80}")
        print("🔬 化合物毒性交互式预测系统")
        print(f"{'=' * 80}")
        print("请选择预测模式:")
        print("1. 单个SMILES预测")
        print("2. 批量CSV文件预测")
        print("输入 'quit' 或 'exit' 退出系统")
        print(f"{'=' * 80}\n")

        while True:
            mode_choice = input("请选择模式 (1/2): ").strip()
            if mode_choice.lower() in ['quit', 'exit', 'q']:
                print("\n👋 感谢使用化合物毒性预测系统!")
                return

            if mode_choice == '1':
                print(f"\n{'=' * 80}")
                print("📝 单个SMILES预测模式")
                print(f"{'=' * 80}")
                print("请输入化合物的SMILES表示进行毒性预测")
                print("输入 'back' 返回上级菜单")
                print("输入 'quit' 或 'exit' 退出系统")
                print(f"{'=' * 80}\n")

                while True:
                    try:
                        smiles_input = input("请输入SMILES: ").strip()
                        if smiles_input.lower() in ['quit', 'exit', 'q']:
                            print("\n👋 感谢使用化合物毒性预测系统!")
                            return
                        if smiles_input.lower() == 'back':
                            print("\n返回主菜单...\n")
                            break
                        if not smiles_input:
                            print("⚠️  请输入有效的SMILES字符串\n")
                            continue

                        result = self.predict_single(smiles_input)
                        if result:
                            print(f"\n{'=' * 60}")
                            print("📊 预测结果")
                            print(f"{'=' * 60}")
                            print(f"规范化SMILES: {result['smiles']}")
                            print(f"毒性预测: {result['prediction']}")
                            print(f"有毒概率: {result['probability']:.4f}")
                            print(f"预测置信度: {result['confidence']:.4f}")
                            print(f"{'=' * 60}\n")
                    except KeyboardInterrupt:
                        print("\n\n👋 程序已终止")
                        return
                    except Exception as e:
                        print(f"❌ 发生错误: {e}\n")

                print(f"\n{'=' * 80}")
                print("请选择预测模式:")
                print("1. 单个SMILES预测")
                print("2. 批量CSV文件预测")
                print("输入 'quit' 或 'exit' 退出系统")
                print(f"{'=' * 80}\n")

            elif mode_choice == '2':
                print(f"\n{'=' * 80}")
                print("📂 批量CSV文件预测模式")
                print(f"{'=' * 80}")
                print("输入 'back' 返回上级菜单")
                print("输入 'quit' 或 'exit' 退出系统")
                print(f"{'=' * 80}\n")

                try:
                    while True:
                        input_csv = input("请输入CSV文件路径: ").strip().strip('"').strip("'")
                        if input_csv.lower() in ['quit', 'exit', 'q']:
                            print("\n👋 感谢使用化合物毒性预测系统!")
                            return
                        if input_csv.lower() == 'back':
                            print("\n返回主菜单...\n")
                            break
                        if not input_csv:
                            print("⚠️  路径不能为空，请重新输入\n")
                            continue
                        if not os.path.exists(input_csv):
                            print(f"❌ 文件不存在: {input_csv}")
                            print(f"提示: 请检查路径是否正确，当前工作目录: {os.getcwd()}")
                            print("请重新输入或输入 'back' 返回上级菜单\n")
                            continue
                        break

                    if input_csv.lower() == 'back':
                        print(f"\n{'=' * 80}")
                        print("请选择预测模式:")
                        print("1. 单个SMILES预测")
                        print("2. 批量CSV文件预测")
                        print("输入 'quit' 或 'exit' 退出系统")
                        print(f"{'=' * 80}\n")
                        continue

                    while True:
                        smiles_col_input = input("请输入SMILES列名 [默认: SMILES]: ").strip()
                        if smiles_col_input.lower() in ['quit', 'exit', 'q']:
                            print("\n👋 感谢使用化合物毒性预测系统!")
                            return
                        if smiles_col_input.lower() == 'back':
                            print("\n返回主菜单...\n")
                            break

                        smiles_col = smiles_col_input or 'SMILES'
                        try:
                            temp_df = pd.read_csv(input_csv)
                            if smiles_col not in temp_df.columns:
                                print(f"❌ 列 '{smiles_col}' 不存在于数据中")
                                print(f"可用列名: {list(temp_df.columns)}")
                                print("请重新输入或输入 'back' 返回上级菜单\n")
                                continue
                            break
                        except Exception as e:
                            print(f"❌ 读取CSV文件失败: {e}")
                            print("请重新输入或输入 'back' 返回上级菜单\n")
                            continue

                    if smiles_col_input.lower() == 'back':
                        print(f"\n{'=' * 80}")
                        print("请选择预测模式:")
                        print("1. 单个SMILES预测")
                        print("2. 批量CSV文件预测")
                        print("输入 'quit' 或 'exit' 退出系统")
                        print(f"{'=' * 80}\n")
                        continue

                    while True:
                        output_csv_input = input(
                            "请输入结果保存路径 [支持目录/文件路径，默认: ./prediction_results.csv]: ").strip().strip(
                            '"').strip("'")
                        if output_csv_input.lower() in ['quit', 'exit', 'q']:
                            print("\n👋 感谢使用化合物毒性预测系统!")
                            return
                        if output_csv_input.lower() == 'back':
                            print("\n返回主菜单...\n")
                            break
                        output_csv = output_csv_input or './prediction_results.csv'
                        break

                    if output_csv_input.lower() == 'back':
                        print(f"\n{'=' * 80}")
                        print("请选择预测模式:")
                        print("1. 单个SMILES预测")
                        print("2. 批量CSV文件预测")
                        print("输入 'quit' 或 'exit' 退出系统")
                        print(f"{'=' * 80}\n")
                        continue

                    result = self.predict_batch_from_csv(input_csv, output_csv, smiles_col)
                    if result is not None:
                        while True:
                            continue_choice = input("\n是否继续预测其他文件? (y/n): ").strip().lower()
                            if continue_choice in ['y', 'yes']:
                                print()
                                break
                            elif continue_choice in ['n', 'no']:
                                print("\n👋 感谢使用化合物毒性预测系统!")
                                return
                            elif continue_choice in ['quit', 'exit', 'q']:
                                print("\n👋 感谢使用化合物毒性预测系统!")
                                return
                            else:
                                print("❌ 无效输入，请输入 y 或 n")
                    else:
                        while True:
                            retry_choice = input("\n预测失败，是否重试? (y/n): ").strip().lower()
                            if retry_choice in ['y', 'yes']:
                                print()
                                break
                            elif retry_choice in ['n', 'no']:
                                print("\n返回主菜单...\n")
                                break
                            elif retry_choice in ['quit', 'exit', 'q']:
                                print("\n👋 感谢使用化合物毒性预测系统!")
                                return
                            else:
                                print("❌ 无效输入，请输入 y 或 n")

                        if retry_choice in ['n', 'no']:
                            print(f"\n{'=' * 80}")
                            print("请选择预测模式:")
                            print("1. 单个SMILES预测")
                            print("2. 批量CSV文件预测")
                            print("输入 'quit' 或 'exit' 退出系统")
                            print(f"{'=' * 80}\n")
                            continue

                except KeyboardInterrupt:
                    print("\n\n👋 程序已终止")
                    return
                except Exception as e:
                    print(f"\n❌ 发生错误: {e}")
                    print("返回主菜单...\n")
                    print(f"\n{'=' * 80}")
                    print("请选择预测模式:")
                    print("1. 单个SMILES预测")
                    print("2. 批量CSV文件预测")
                    print("输入 'quit' 或 'exit' 退出系统")
                    print(f"{'=' * 80}\n")
            else:
                print("❌ 无效的选项，请选择 1 或 2\n")


def main():
    """主程序流程"""
    print("=" * 80)
    print("         化合物毒性预测系统系统 V1.0         ")
    print("    Compound Toxicity Prediction Platform    ")
    print("=" * 80)

    print("\n请选择操作模式:")
    print("1. 训练新模型")
    print("2. 加载已有模型并预测")

    choice = input("\n请输入选项 (1/2): ").strip()

    if choice == '1':
        print("\n" + "=" * 80)
        print("训练新模型模式")
        print("=" * 80)

        print("\n📁 步骤1: 设置模型保存路径")
        while True:
            model_save_dir = input("请输入模型保存目录路径 [默认: ./trained_models]: ").strip()
            if model_save_dir.lower() in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            if not model_save_dir:
                model_save_dir = './trained_models'
                break
            try:
                os.makedirs(model_save_dir, exist_ok=True)
                break
            except Exception as e:
                print(f"❌ 无法创建目录: {e}")
                print("请重新输入或输入 'quit' 退出\n")

        platform = ToxicityPredictionPlatform(model_save_dir=model_save_dir)
        print(f"✅ 模型将保存至: {os.path.abspath(model_save_dir)}")

        print("\n📊 步骤2: 加载训练数据")
        print("请输入CSV文件路径（文件需包含SMILES列和标签列）")
        while True:
            csv_file = input("CSV文件路径: ").strip()
            if csv_file.lower() in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            csv_file = csv_file.strip('"').strip("'")
            if not csv_file:
                print("⚠️  路径不能为空，请重新输入\n")
                continue
            if not os.path.exists(csv_file):
                print(f"❌ 文件不存在: {csv_file}")
                print(f"提示: 请检查路径是否正确，当前工作目录为: {os.getcwd()}")
                print("请重新输入或输入 'quit' 退出\n")
                continue
            break

        print("\n请指定数据列名 (直接回车使用默认值):")
        while True:
            smiles_col_input = input("SMILES列名 [默认: SMILES]: ").strip()
            if smiles_col_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            smiles_col = smiles_col_input or 'SMILES'

            label_col_input = input("标签列名 [默认: Toxicity_Label]: ").strip()
            if label_col_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            label_col = label_col_input or 'Toxicity_Label'

            try:
                temp_df = pd.read_csv(csv_file)
                missing_cols = []
                if smiles_col not in temp_df.columns:
                    missing_cols.append(smiles_col)
                if label_col not in temp_df.columns:
                    missing_cols.append(label_col)
                if missing_cols:
                    print(f"❌ 以下列不存在于数据中: {missing_cols}")
                    print(f"可用列名: {list(temp_df.columns)}")
                    print("请重新输入或输入 'quit' 退出\n")
                    continue
                break
            except Exception as e:
                print(f"❌ 读取CSV文件失败: {e}")
                print("请重新输入或输入 'quit' 退出\n")
                continue

        graphs = platform.load_and_preprocess_data(csv_file, smiles_col=smiles_col, label_col=label_col)
        if len(graphs) == 0:
            print("❌ 没有成功加载任何图数据，请检查数据格式")
            return

        print("\n⚙️  步骤3: 设置训练参数")
        print("=" * 60)
        print("以下参数可自定义，直接回车使用默认值")
        print("=" * 60)

        print("\n【训练模式选择】")
        train_mode_input = input("  训练模式 (cv=交叉验证/standard=普通训练) [默认: cv]: ").strip().lower() or 'cv'
        while train_mode_input not in ['cv', 'standard']:
            print("  ❌ 无效模式，仅支持 cv 或 standard")
            train_mode_input = input("  训练模式 (cv=交叉验证/standard=普通训练) [默认: cv]: ").strip().lower() or 'cv'

        try:
            if train_mode_input == 'cv':
                print("\n【交叉验证参数】")
                n_folds_input = input("  交叉验证折数 (2-10) [默认: 5]: ").strip()
                n_folds = int(n_folds_input) if n_folds_input else 5
                if not (2 <= n_folds <= 10):
                    print("  ⚠️  折数超出范围，使用默认值5")
                    n_folds = 5
                test_size, random_state = 0.2, 42
            else:
                print("\n【普通训练参数】")
                test_size_input = input("  验证集比例 (0-1之间) [默认: 0.2]: ").strip()
                test_size = float(test_size_input) if test_size_input else 0.2
                random_state = int(input("  随机种子 [默认: 42]: ") or "42")
                n_folds = 5

            print("\n【通用训练参数】")
            n_epochs_input = input("  训练轮数 (5-5000) [默认: 200]: ").strip()
            n_epochs = int(n_epochs_input) if n_epochs_input else 200
            if not (5 <= n_epochs <= 5000):
                print("  ⚠️  训练轮数超出范围，使用默认值200")
                n_epochs = 200

            batch_size_input = input("  批次大小 (16-1024) [默认: 64]: ").strip()
            batch_size = int(batch_size_input) if batch_size_input else 64
            if not (16 <= batch_size <= 1024):
                print("  ⚠️  批次大小超出范围，使用默认值64")
                batch_size = 64

            print("\n【模型架构参数】")
            num_layers_input = input("  GCN层数 (1-10) [默认: 2]: ").strip()
            num_layers = int(num_layers_input) if num_layers_input else 2
            if not (1 <= num_layers <= 10):
                print("  ⚠️  层数超出范围，使用默认值2")
                num_layers = 2

            hidden_dim_input = input("  隐藏层维度 (16-10000) [默认: 128]: ").strip()
            hidden_dim = int(hidden_dim_input) if hidden_dim_input else 128
            if not (16 <= hidden_dim <= 10000):
                print("  ⚠️  隐藏层维度超出范围，使用默认值128")
                hidden_dim = 128

            dropout_input = input("  Dropout比率 (0.1-1) [默认: 0.3]: ").strip()
            dropout = float(dropout_input) if dropout_input else 0.3
            if not (0.1 <= dropout <= 1.0):
                print("  ⚠️  Dropout比率超出范围，使用默认值0.3")
                dropout = 0.3

            print("\n【优化器参数】")
            lr_input = input("  学习率 (0.00001-0.1) [默认: 0.0001]: ").strip()
            learning_rate = float(lr_input) if lr_input else 0.0001
            if not (0.00001 <= learning_rate <= 0.1):
                print("  ⚠️  学习率超出范围，使用默认值0.0001")
                learning_rate = 0.0001

            wd_input = input("  权重衰减 (1e-8到1e-1) [默认: 1e-4]: ").strip()
            weight_decay = float(wd_input) if wd_input else 1e-4
            if not (1e-8 <= weight_decay <= 1e-1):
                print("  ⚠️  权重衰减超出范围，使用默认值1e-4")
                weight_decay = 1e-4

            print("\n【损失函数参数】")
            loss_type_input = input("  损失函数 (focal/bce) [默认: focal]: ").strip().lower() or 'focal'
            if loss_type_input not in ['focal', 'bce']:
                print("  ⚠️  损失函数无效，使用默认值 focal")
                loss_type_input = 'focal'

            focal_alpha_input = input("  Focal alpha [默认: 0.85]: ").strip()
            focal_alpha = float(focal_alpha_input) if focal_alpha_input else 0.85

            focal_gamma_input = input("  Focal gamma [默认: 1.7]: ").strip()
            focal_gamma = float(focal_gamma_input) if focal_gamma_input else 1.7

        except ValueError:
            print("\n⚠️  输入无效,使用默认参数")
            train_mode_input = 'cv'
            n_folds, test_size, random_state = 5, 0.2, 42
            n_epochs, batch_size = 200, 64
            num_layers, hidden_dim, dropout = 2, 128, 0.3
            learning_rate, weight_decay = 0.0001, 1e-4
            loss_type_input, focal_alpha, focal_gamma = 'focal', 0.85, 1.7

        print(f"\n{'=' * 60}")
        print("训练配置总结:")
        print(f"{'=' * 60}")
        print(f"【训练模式】")
        print(f"  - 模式: {train_mode_input}")
        if train_mode_input == 'cv':
            print(f"  - 交叉验证折数: {n_folds}")
        else:
            print(f"  - 验证集比例: {test_size}")
            print(f"  - 随机种子: {random_state}")
        print("【通用训练】")
        print(f"  - 训练轮数: {n_epochs}")
        print(f"  - 批次大小: {batch_size}")
        print("【模型架构】")
        print(f"  - GCN层数: {num_layers}")
        print(f"  - 隐藏层维度: {hidden_dim}")
        print(f"  - Dropout比率: {dropout}")
        print("【优化器】")
        print(f"  - 学习率: {learning_rate}")
        print(f"  - 权重衰减: {weight_decay}")
        print("【损失函数】")
        print(f"  - 类型: {loss_type_input}")
        print(f"  - Focal alpha: {focal_alpha}")
        print(f"  - Focal gamma: {focal_gamma}")
        print(f"{'=' * 60}\n")

        cv_tracker = platform.train(
            graphs,
            training_mode=train_mode_input,
            n_folds=n_folds,
            test_size=test_size,
            random_state=random_state,
            n_epochs=n_epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_layers=num_layers,
            loss_type=loss_type_input,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

        print("\n💾 步骤4: 保存训练好的模型")
        while True:
            save_choice = input("是否保存模型? (y/n) [默认: y]: ").strip().lower() or 'y'
            if save_choice in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            if save_choice in ['y', 'yes']:
                while True:
                    model_name = input("请输入模型名称 (不含.pth后缀，直接回车使用时间戳): ").strip()
                    if model_name.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 程序已退出")
                        return
                    try:
                        platform.save_model(model_name if model_name else None)
                        print(f"提示: 模型已保存在 {os.path.abspath(model_save_dir)} 目录下")
                        break
                    except Exception as e:
                        print(f"❌ 保存失败: {e}")
                        print("请重新输入模型名称或输入 'quit' 退出\n")
                break
            elif save_choice in ['n', 'no']:
                print("跳过保存模型")
                break
            else:
                print("❌ 无效输入，请输入 y 或 n\n")

        print("\n🔬 步骤5: 使用训练好的模型进行预测")
        while True:
            predict_choice = input("是否立即进行预测? (y/n) [默认: y]: ").strip().lower() or 'y'
            if predict_choice in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            if predict_choice in ['y', 'yes']:
                platform.interactive_prediction()
                break
            elif predict_choice in ['n', 'no']:
                print("\n👋 感谢使用化合物毒性预测系统!")
                break
            else:
                print("❌ 无效输入，请输入 y 或 n\n")

    elif choice == '2':
        print("\n" + "=" * 80)
        print("加载已有模型并预测模式")
        print("=" * 80)

        platform = ToxicityPredictionPlatform(model_save_dir='./trained_models')

        print("\n请输入要加载的模型文件路径 (.pth文件)")
        while True:
            model_path = input("模型文件路径: ").strip()
            if model_path.lower() in ['quit', 'exit', 'q']:
                print("\n👋 程序已退出")
                return
            model_path = model_path.strip('"').strip("'")
            if not model_path:
                print("⚠️  路径不能为空，请重新输入\n")
                continue
            if not os.path.exists(model_path):
                print(f"❌ 文件不存在: {model_path}")
                print(f"提示: 请检查路径是否正确，当前工作目录为: {os.getcwd()}")
                print("请重新输入或输入 'quit' 退出\n")
                continue
            try:
                platform.load_model(model_path)
                break
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print("请重新输入或输入 'quit' 退出\n")
                continue

        platform.interactive_prediction()

    else:
        print("\n❌ 无效的选项，请选择 1 或 2")
        print("程序已退出")


if __name__ == "__main__":
    main()
