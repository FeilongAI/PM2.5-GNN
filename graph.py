import os  # 导入操作系统模块
import sys  # 导入系统模块
proj_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径并获取其目录
sys.path.append(proj_dir)  # 将项目目录添加到系统路径中
import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
from collections import OrderedDict  # 从collections模块导入OrderedDict类
from scipy.spatial import distance  # 从scipy.spatial模块导入distance函数
from torch_geometric.utils import dense_to_sparse, to_dense_adj  # 从torch_geometric.utils模块导入dense_to_sparse和to_dense_adj函数
from geopy.distance import geodesic  # 从geopy.distance模块导入geodesic函数
from metpy.units import units  # 从metpy.units模块导入units
import metpy.calc as mpcalc  # 导入metpy.calc模块并重命名为mpcalc
from bresenham import bresenham  # 从bresenham模块导入bresenham函数

city_fp = os.path.join(proj_dir, '/root/pm25/data/city.txt')  # 获取city.txt文件的路径
altitude_fp = os.path.join(proj_dir, '/root/pm25/data/altitude.npy')  # 获取altitude.npy文件的路径

class Graph():  # 定义Graph类
    def __init__(self):  # 初始化方法
        self.dist_thres = 3  # 设置距离阈值为3
        self.alti_thres = 1200  # 设置海拔阈值为1200
        self.use_altitude = True  # 设置是否使用海拔为True

        self.altitude = self._load_altitude()  # 加载海拔数据
        self.nodes = self._gen_nodes()  # 生成节点
        self.node_attr = self._add_node_attr()  # 添加节点属性
        self.node_num = len(self.nodes)  # 获取节点数量
        self.edge_index, self.edge_attr = self._gen_edges()  # 生成边和边属性
        if self.use_altitude:  # 如果使用海拔
            self._update_edges()  # 更新边
        self.edge_num = self.edge_index.shape[1]  # 获取边数量
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]  # 将边索引转换为稠密邻接矩阵

    def _load_altitude(self):  # 定义加载海拔数据的方法
        assert os.path.isfile(altitude_fp)  # 断言海拔文件存在
        altitude = np.load(altitude_fp)  # 加载海拔数据
        return altitude  # 返回海拔数据(641, 561)

    def _lonlat2xy(self, lon, lat, is_aliti):  # 定义经纬度转换为xy坐标的方法
        if is_aliti:  # 如果是海拔
            lon_l = 100.0  # 设置左经度为100.0
            lon_r = 128.0  # 设置右经度为128.0
            lat_u = 48.0  # 设置上纬度为48.0
            lat_d = 16.0  # 设置下纬度为16.0
            res = 0.05  # 设置分辨率为0.05
        else:  # 如果不是海拔
            lon_l = 103.0  # 设置左经度为103.0
            lon_r = 122.0  # 设置右经度为122.0
            lat_u = 42.0  # 设置上纬度为42.0
            lat_d = 28.0  # 设置下纬度为28.0
            res = 0.125  # 设置分辨率为0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))  # 计算x坐标
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))  # 计算y坐标
        return x, y  # 返回x和y坐标

    def _gen_nodes(self):  # 定义生成节点的方法
        nodes = OrderedDict()  # 创建有序字典存储节点
        with open(city_fp, 'r') as f:  # 打开城市文件
            for line in f:  # 遍历文件中的每一行
                idx, city, lon, lat = line.rstrip('\n').split(' ')  # 解析行数据
                idx = int(idx)  # 将索引转换为整数
                lon, lat = float(lon), float(lat)  # 将经纬度转换为浮点数
                x, y = self._lonlat2xy(lon, lat, True)  # 将经纬度转换为xy坐标
                altitude = self.altitude[y, x]  # 获取海拔数据
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})  # 更新节点字典{'altitude': 18.0, 'city': 'Beijing', 'lat': 40.045975000000006, 'lon': 116.39824999999998}
        return nodes  # 返回节点字典

    def _add_node_attr(self):  # 定义添加节点属性的方法
        node_attr = []  # 创建空列表存储节点属性
        altitude_arr = []  # 创建空列表存储海拔数据
        for i in self.nodes:  # 遍历节点
            altitude = self.nodes[i]['altitude']  # 获取节点的海拔数据
            altitude_arr.append(altitude)  # 将海拔数据添加到列表中
        altitude_arr = np.stack(altitude_arr)  # 将海拔数据堆叠成数组
        node_attr = np.stack([altitude_arr], axis=-1)  # 将海拔数据堆叠成节点属性数组
        return node_attr  # 返回节点属性数组

    def traverse_graph(self):  # 定义遍历图的方法
        lons = []  # 创建空列表存储经度
        lats = []  # 创建空列表存储纬度
        citys = []  # 创建空列表存储城市
        idx = []  # 创建空列表存储索引
        for i in self.nodes:  # 遍历节点
            idx.append(i)  # 将索引添加到列表中
            city = self.nodes[i]['city']  # 获取城市名称
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']  # 获取经纬度
            lons.append(lon)  # 将经度添加到列表中
            lats.append(lat)  # 将纬度添加到列表中
            citys.append(city)  # 将城市名称添加到列表中
        return idx, citys, lons, lats  # 返回索引、城市名称、经度和纬度列表

    def gen_lines(self):  # 定义生成线段的方法
        lines = []  # 创建空列表存储线段
        for i in range(self.edge_index.shape[1]):  # 遍历边索引
            src, dest = self.edge_index[0, i], self.edge_index[1, i]  # 获取源节点和目标节点
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']  # 获取源节点的经纬度
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']  # 获取目标节点的经纬度
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))  # 将线段添加到列表中
        return lines  # 返回线段列表

    def _gen_edges(self):  # 定义生成边的方法
        coords = []  # 创建空列表存储坐标
        lonlat = {}  # 创建空字典存储经纬度
        for i in self.nodes:  # 遍历节点
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])  # 将节点的经纬度添加到坐标列表中
        dist = distance.cdist(coords, coords, 'euclidean')  # 计算坐标之间的欧几里得距离
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)  # 创建零矩阵作为邻接矩阵(184, 184)
        adj[dist <= self.dist_thres] = 1  # 将距离小于等于阈值的元素设置为1
        assert adj.shape == dist.shape  # 断言邻接矩阵的形状与距离矩阵相同
        dist = dist * adj  # 将距离矩阵与邻接矩阵相乘 获得邻接矩阵节点的距离
        edge_index, dist = dense_to_sparse(torch.tensor(dist))#边和边之间的链接 边和边之间的属性即距离  # 将稠密矩阵转换为稀疏矩阵
        edge_index, dist = edge_index.numpy(), dist.numpy()  # 将边索引和距离转换为NumPy数组

        direc_arr = []  # 计算与想关城市的风向创建空列表存储方向
        dist_kilometer = []  # 创建空列表存储距离（公里）
        for i in range(edge_index.shape[1]):  # 遍历边索引
            src, dest = edge_index[0, i], edge_index[1, i]  # 获取源节点和目标节点
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']  # 获取源节点的经纬度
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']  # 获取目标节点的经纬度
            src_location = (src_lat, src_lon)  # 创建源节点位置元组
            dest_location = (dest_lat, dest_lon)  # 创建目标节点位置元组
            dist_km = geodesic(src_location, dest_location).kilometers  # 计算源节点和目标节点之间的距离（公里）
            v, u = src_lat - dest_lat, src_lon - dest_lon  # 计算纬度和经度的差值

            u = u * units.meter / units.second  # 将经度差值转换为速度单位
            v = v * units.meter / units.second  # 将纬度差值转换为速度单位
            direc = mpcalc.wind_direction(u, v)._magnitude  # 计算风向

            direc_arr.append(direc)  # 将风向添加到列表中
            dist_kilometer.append(dist_km)  # 将距离（公里）添加到列表中

        direc_arr = np.stack(direc_arr)  # 将风向列表堆叠成数组
        dist_arr = np.stack(dist_kilometer)  # 将距离（公里）列表堆叠成数组
        attr = np.stack([dist_arr, direc_arr], axis=-1)  # 将距离和风向堆叠成边属性数组

        return edge_index, attr  # 返回边索引和边属性包括风向和距离

    def _update_edges(self):  # 定义更新边的方法
        edge_index = []  # 创建空列表存储边索引
        edge_attr = []  # 创建空列表存储边属性
        for i in range(self.edge_index.shape[1]):  # 遍历边索引
            src, dest = self.edge_index[0, i], self.edge_index[1, i]  # 获取源节点和目标节点
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']  # 获取源节点的经纬度
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']  # 获取目标节点的经纬度
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)  # 将源节点的经纬度转换为xy坐标
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)  # 将目标节点的经纬度转换为xy坐标
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))  # 使用Bresenham算法生成线段上的点
            altitude_points = self.altitude[points[0], points[1]]  # 获取线段上点的海拔数据
            altitude_src = self.altitude[src_y, src_x]  # 获取源节点的海拔数据
            altitude_dest = self.altitude[dest_y, dest_x]  # 获取目标节点的海拔数据
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:  # 判断海拔差异是否小于阈值
                edge_index.append(self.edge_index[:,i])  # 将边索引添加到列表中
                edge_attr.append(self.edge_attr[i])  # 将边属性添加到列表中

        self.edge_index = np.stack(edge_index, axis=1)  # 将边索引堆叠成数组
        self.edge_attr = np.stack(edge_attr, axis=0)  # 将边属性堆叠成数组

if __name__ == '__main__':  # 主程序入口
    graph = Graph()  # 创建Graph对象