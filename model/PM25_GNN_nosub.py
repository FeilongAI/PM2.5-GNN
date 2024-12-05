import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter
from model.iTransformer import Model

class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)#边的二维邻接矩阵
        self.edge_attr = torch.Tensor(np.float32(edge_attr))#边风速
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)#这行代码对 self.edge_attr 进行了标准化处理。具体来说，它将每个特征减去其均值，然后除以其标准差，使得每个特征的分布转换为标准正态分布（均值为0，标准差为1）。这有助于加速模型的训练并提高模型的性能。
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))#模型参数
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)#风的平均风速
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)#风的标准差
        e_h = 32  # edge MLP hidden layer dimension
        e_out = 30  # edge MLP output dimension
        n_out = out_dim  # node MLP output dimension
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)  # 将 edge_index 转移到设备上
        self.edge_attr = self.edge_attr.to(self.device)  # 将 edge_attr 转移到设备上
        self.w = self.w.to(self.device)  # 将参数 w 转移到设备上
        self.b = self.b.to(self.device)  # 将参数 b 转移到设备上

        edge_src, edge_target = self.edge_index  # 获取边的源节点和目标节点索引
        node_src = x[:, edge_src]  # 获取源节点的特征
        node_target = x[:, edge_target]  # 获取目标节点的特征

        src_wind = node_src[:, :, -2:] * self.wind_std[None, None, :] + self.wind_mean[None, None, :]  # 计算源节点的风速和风向
        src_wind_speed = src_wind[:, :, 0]  # 获取源节点的风速
        src_wind_direc = src_wind[:, :, 1]  # 获取源节点的风向
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)  # 扩展边特征以匹配源节点的批次大小
        city_dist = self.edge_attr_[:, :, 0]  # 获取城市距离
        city_direc = self.edge_attr_[:, :, 1]  # 获取城市方向

        theta = torch.abs(city_direc - src_wind_direc)  # 计算风向差异
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)  # 计算边权重
        edge_weight = edge_weight.to(self.device)  # 将边权重转移到设备上
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)  # 标准化边特征并转移到设备上
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:, :, None]], dim=-1)  # 将源节点、目标节点、标准化边特征和边权重拼接在一起

        out = self.edge_mlp(out)  # 通过边 MLP 处理
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))  # 聚合边特征这行代码使用 torch_scatter 库中的 scatter_add 函数来聚合边特征。具体来说，它将 out 张量中的值根据 edge_target 索引沿指定维度 dim 进行加和，并将结果存储在一个新的张量中。dim_size 参数指定了输出张量在 dim 维度上的大小
        #out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))

        out = out_add  # + out_sub  # 将聚合结果赋值给 out torch.Size([32, 184, 30])
        out = self.node_mlp(out)  # 通过节点 MLP 处理

        return out  # 返回最终输出torch.Size([32, 184, 13])


class PM25_GNN_nosub(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(PM25_GNN_nosub, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim#嵌入维度
        self.hid_dim = 64#隐藏层维度
        self.out_dim = 1#输出维度
        self.gnn_out = 13#gnn输出维度

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)#全连接层
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):#pm25_hist torch.Size([32, 1, 184, 1]) batch_size seq_len city_num, pm25 feature torch.Size([32, 25, 184, 12])
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]#torch.Size([32, 1, 184, 1]) 总是选择pm25_hist 的最后一个时间步，无论第二维度的大小是多少。xn：torch.Size([32, 184, 1])
        for i in range(self.pred_len):  # 遍历预测长度
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)#选择最后一个时间步的特征按照最后一个维度拼接在一起，x,feature包含未来的特征？

            xn_gnn = x #torch.Size([32, 184, 13])
            xn_gnn = xn_gnn.contiguous()#保证在内存中连续
            xn_gnn = self.graph_gnn(xn_gnn)  # 通过GraphGNN处理，torch.Size([32, 184, 13])
            x = torch.cat([xn_gnn, x], dim=-1)  # 将GraphGNN的输出与原始张量拼接，torch.Size([32, 184, 26])

            hn = self.gru_cell(x, hn)  # 使用GRU单元更新隐藏状态
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)  # 将隐藏状态重塑为匹配批次和城市维度
            xn = self.fc_out(xn)  # torch.Size([32, 184, 1])通过全连接层得到预测值这里使用预测值去继续预测下一个
            pm25_pred.append(xn)  # 将预测值添加到列表中

        pm25_pred = torch.stack(pm25_pred, dim=1)  # torch.Size([32, 24, 184, 1])沿时间维度堆叠预测值

        return pm25_pred  # torch.Size([32, 24, 184, 1])返回最终预测值
